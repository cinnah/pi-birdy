"""Camera loop: lores motion detection, full-res still on trigger. Pi only."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from birdy.config import Config
from birdy.daylight import DaylightGate
from birdy.live import LiveConfig
from birdy.motion import MotionDetector
from birdy.pipeline import MotionEvent, Pipeline
from birdy.state import RuntimeState
from birdy.store import Store

log = logging.getLogger(__name__)

FRAME_INTERVAL = 0.1  # ~10 fps analysis
GATE_RECHECK = 60.0
MAX_CONSECUTIVE_FRAME_ERRORS = 50  # ~5 s at FRAME_INTERVAL
# False triggers at a 3 s cooldown can write several GB/day and non-birds are kept
# for 7 days, so stop capturing before the card fills rather than wedge the Pi.
MIN_FREE_BYTES = 500 * 1024 * 1024

_low_disk_warned = False  # reset at each camera session start


@contextmanager
def camera_session(config: Config):
    from picamera2 import Picamera2  # Pi only

    picam2 = Picamera2()
    try:
        # raw=None: no raw stream — we never read it and it costs ~10 MB of CMA per buffer.
        cam_cfg = picam2.create_preview_configuration(
            main={"size": tuple(config.camera.main_size), "format": "RGB888"},
            lores={"size": tuple(config.camera.lores_size), "format": "YUV420"},
            raw=None,
            buffer_count=2,
        )
        picam2.configure(cam_cfg)
        picam2.options["quality"] = config.camera.jpeg_quality
        picam2.start()
        time.sleep(1.0)  # let AE/AWB settle
        yield picam2
    finally:
        # close() stops if started; wrapping configure/start too means a failed
        # session start never leaves the camera acquired.
        picam2.close()


def lores_grey(picam2, lores_size: tuple[int, int]):
    """Y plane of the lores stream as an (h, w) uint8 array, stride-safe."""
    w, h = lores_size
    stride = picam2.stream_configuration("lores")["stride"]
    buf = picam2.capture_buffer("lores")
    # ascontiguousarray: the [:, :w] slice is strided and OpenCV wants a contiguous input.
    return np.ascontiguousarray(buf[: stride * h].reshape(h, stride)[:, :w])


def grab_lores_frame(config: Config):
    """One-shot helper for `birdy roi`."""
    with camera_session(config) as picam2:
        return lores_grey(picam2, config.camera.lores_size)


def _capture_still(picam2, store: Store, pipeline: Pipeline, score: int) -> None:
    global _low_disk_warned
    free = store.disk_free_bytes()
    if free < MIN_FREE_BYTES:
        if not _low_disk_warned:
            log.warning("disk free %d MB < %d MB; skipping captures", free // 2**20, MIN_FREE_BYTES // 2**20)
            _low_disk_warned = True
        return
    now = datetime.now().astimezone()
    abs_path, rel_path = store.photo_path_for(now)
    picam2.capture_file(str(abs_path))
    capture_id = store.add_capture(now, rel_path, score)
    pipeline.submit(MotionEvent(capture_id=capture_id, path=abs_path, captured_at=now, motion_score=score))
    log.info("captured %s (motion %d)", rel_path, score)


def _tunable_key(cfg: Config) -> tuple:
    """The config fields the detector and gate are built from (not site.name)."""
    return (cfg.motion, cfg.site.lat, cfg.site.lon, cfg.daylight)


def _build_tunables(cfg: Config, lores_size: tuple[int, int]) -> tuple[MotionDetector, DaylightGate]:
    # `lores_size` is the process's pinned camera size, not cfg.camera.lores_size:
    # the detector must match the stream the running session actually delivers.
    # MotionDetector's default warmup_frames (10) suppresses triggers while
    # auto-exposure settles after the camera session starts, and again for ~1 s
    # after a settings change rebuilds the detector.
    detector = MotionDetector(
        frame_size=lores_size,
        roi=cfg.motion.roi,
        pixel_threshold=cfg.motion.pixel_threshold,
        min_area=cfg.motion.min_area,
        background_alpha=cfg.motion.background_alpha,
        cooldown_seconds=cfg.motion.cooldown_seconds,
    )
    gate = DaylightGate(cfg.site.lat, cfg.site.lon, cfg.daylight.margin_minutes)
    return detector, gate


def run_capture(
    live: LiveConfig,
    store: Store,
    pipeline: Pipeline,
    state: RuntimeState,
    stop: threading.Event,
) -> None:
    global _low_disk_warned
    paused_logged = False
    # Camera settings are fixed for the process — the classifier's crop scaling was
    # built from them too; the watcher warns that a restart is required.
    camera = live.get().camera
    version = live.version
    config = live.get()
    detector, gate = _build_tunables(config, camera.lores_size)
    built_from = _tunable_key(config)

    def refresh_tunables() -> bool:
        """Pick up a settings change; True if the detector and gate were rebuilt."""
        nonlocal version, detector, gate, built_from
        if live.version == version:
            return False
        version = live.version
        cfg = live.get()
        key = _tunable_key(cfg)
        if key == built_from:
            # Only classifier/retention/site name etc. changed: keep the warmed-up detector.
            log.info("settings applied (v%d)", version)
            return False
        detector, gate = _build_tunables(cfg, camera.lores_size)
        built_from = key
        log.info("settings applied (v%d) (detector rebuilt)", version)
        return True

    while not stop.is_set():
        refresh_tunables()
        override_version = state.override_version
        if not state.camera_wanted(gate, datetime.now().astimezone()):
            manual = state.override == "off"
            state.update(state="manual_off" if manual else "daylight_paused", motion_score=0)
            if not paused_logged:
                if manual:
                    log.info("camera switched off from the web page until %s", state.override_until)
                else:
                    log.info("outside daylight window; next change %s", gate.next_change(datetime.now().astimezone()))
                paused_logged = True
            # Wait in short steps so a lat/lon/margin change or the camera switch un-pauses within ~1 s.
            deadline = time.monotonic() + GATE_RECHECK
            while (
                not stop.is_set()
                and live.version == version
                and state.override_version == override_version
                and time.monotonic() < deadline
            ):
                stop.wait(min(1.0, max(0.0, deadline - time.monotonic())))
            if state.override_version != override_version:
                paused_logged = False  # log the new reason, if still paused
            continue
        if paused_logged or state.override == "on":
            log.info("starting camera%s", " (switched on from the web page)" if state.override == "on" else "")
            paused_logged = False

        # Read version before the snapshot: a replace() landing between the two
        # reads then triggers a (redundant) rebuild rather than being skipped.
        # A fresh detector per session: its background model is stale after a pause.
        version = live.version
        config = dataclasses.replace(live.get(), camera=camera)
        detector, gate = _build_tunables(config, camera.lores_size)
        built_from = _tunable_key(config)
        try:
            with camera_session(config) as picam2:
                state.update(state="capturing")
                _low_disk_warned = False
                last_gate_check = time.monotonic()
                override_version = state.override_version
                consecutive_errors = 0
                while not stop.is_set():
                    if refresh_tunables() or state.override_version != override_version:
                        # new gate or camera switch: re-check it now, not in up to 60 s
                        last_gate_check = float("-inf")
                        override_version = state.override_version
                    if time.monotonic() - last_gate_check > GATE_RECHECK:
                        last_gate_check = time.monotonic()
                        if not state.camera_wanted(gate, datetime.now().astimezone()):
                            break
                    t0 = time.monotonic()
                    try:
                        frame = lores_grey(picam2, config.camera.lores_size)
                        result = detector.process(frame)
                        # No copy needed: capture_buffer returns a fresh array each call and
                        # RuntimeState.lores_frame() copies on read.
                        state.update(motion_score=result.score, latest_lores=frame)
                        if result.triggered:
                            _capture_still(picam2, store, pipeline, result.score)
                        consecutive_errors = 0
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                            log.error("%d consecutive frame failures; giving up on this camera session", consecutive_errors)
                            raise RuntimeError("camera stopped delivering frames")
                        log.exception("frame processing failed; continuing")
                    elapsed = time.monotonic() - t0
                    if elapsed < FRAME_INTERVAL:
                        stop.wait(FRAME_INTERVAL - elapsed)
        except Exception:
            state.update(state="camera_error")
            log.exception("camera session failed")
            raise
