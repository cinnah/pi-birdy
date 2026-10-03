"""Camera loop: lores motion detection, full-res still on trigger. Pi only."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from birdy.config import Config
from birdy.daylight import DaylightGate
from birdy.motion import MotionDetector
from birdy.pipeline import MotionEvent, Pipeline
from birdy.state import RuntimeState
from birdy.store import Store

log = logging.getLogger(__name__)

FRAME_INTERVAL = 0.1  # ~10 fps analysis
GATE_RECHECK = 60.0
MAX_CONSECUTIVE_FRAME_ERRORS = 50  # ~5 s at FRAME_INTERVAL


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


def _capture_still(picam2, store: Store, pipeline: Pipeline, state: RuntimeState, score: int) -> None:
    now = datetime.now().astimezone()
    abs_path, rel_path = store.photo_path_for(now)
    picam2.capture_file(str(abs_path))
    capture_id = store.add_capture(now, rel_path, score)
    state.update(last_capture_at=now)
    pipeline.submit(MotionEvent(capture_id=capture_id, path=abs_path, captured_at=now, motion_score=score))
    log.info("captured %s (motion %d)", rel_path, score)


def run_capture(
    config: Config,
    store: Store,
    pipeline: Pipeline,
    state: RuntimeState,
    gate: DaylightGate,
    stop: threading.Event,
) -> None:
    paused_logged = False
    while not stop.is_set():
        if not gate.is_active(datetime.now().astimezone()):
            state.update(state="daylight_paused", motion_score=0)
            if not paused_logged:
                log.info("outside daylight window; next change %s", gate.next_change(datetime.now().astimezone()))
                paused_logged = True
            stop.wait(GATE_RECHECK)
            continue
        if paused_logged:
            log.info("daylight window open; starting camera")
            paused_logged = False

        # MotionDetector's default warmup_frames (10) suppresses triggers while
        # auto-exposure settles after the camera session starts.
        detector = MotionDetector(
            frame_size=config.camera.lores_size,
            roi=config.motion.roi,
            pixel_threshold=config.motion.pixel_threshold,
            min_area=config.motion.min_area,
            background_alpha=config.motion.background_alpha,
            cooldown_seconds=config.motion.cooldown_seconds,
        )
        try:
            with camera_session(config) as picam2:
                state.update(state="capturing")
                last_gate_check = time.monotonic()
                consecutive_errors = 0
                while not stop.is_set():
                    if time.monotonic() - last_gate_check > GATE_RECHECK:
                        last_gate_check = time.monotonic()
                        if not gate.is_active(datetime.now().astimezone()):
                            break
                    t0 = time.monotonic()
                    try:
                        frame = lores_grey(picam2, config.camera.lores_size)
                        result = detector.process(frame)
                        # No copy needed: capture_buffer returns a fresh array each call and
                        # RuntimeState.lores_frame() copies on read.
                        state.update(motion_score=result.score, latest_lores=frame)
                        if result.triggered:
                            _capture_still(picam2, store, pipeline, state, result.score)
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
