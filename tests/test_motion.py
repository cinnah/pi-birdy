import numpy as np

from birdy.motion import BBox, MotionDetector, roi_bbox

W, H = 320, 240
ROI = ((40, 60), (280, 60), (280, 220), (40, 220))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_detector(clock, **overrides):
    kw = dict(frame_size=(W, H), roi=ROI, pixel_threshold=25, min_area=150, background_alpha=0.05, cooldown_seconds=3.0, clock=clock)
    kw.update(overrides)
    return MotionDetector(**kw)


def blank():
    return np.full((H, W), 50, dtype=np.uint8)


def with_blob(x, y, size=30):
    f = blank()
    f[y:y + size, x:x + size] = 255
    return f


def test_first_frame_initialises_background_without_trigger():
    d = make_detector(FakeClock())
    r = d.process(blank())
    assert r.score == 0 and r.triggered is False


def test_static_scene_never_triggers():
    d = make_detector(FakeClock())
    for _ in range(20):
        r = d.process(blank())
    assert r.score == 0 and r.triggered is False


def test_blob_inside_roi_triggers_with_score():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100))
    assert r.triggered is True
    # 30x30 blob; the 5x5 blur pushes a 1px halo over the threshold, so the
    # changed region is ~32x32 minus corners (measured 1020 on OpenCV 4.x).
    assert 970 <= r.score <= 1070


def test_blob_outside_roi_does_not_trigger():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(0, 0))
    assert r.score == 0 and r.triggered is False


def test_cooldown_blocks_second_trigger():
    clock = FakeClock()
    d = make_detector(clock)
    d.process(blank())
    assert d.process(with_blob(100, 100)).triggered
    clock.t = 1.0
    assert d.process(with_blob(150, 150)).triggered is False
    clock.t = 3.5
    assert d.process(with_blob(200, 100)).triggered is True


def test_stationary_blob_fades_into_background():
    clock = FakeClock()
    d = make_detector(clock)
    d.process(blank())
    for i in range(100):
        clock.t = i * 10.0
        r = d.process(with_blob(100, 100))
    assert r.score == 0


def test_small_blob_below_min_area_does_not_trigger():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100, size=5))
    assert 0 < r.score < 150 and r.triggered is False


def test_roi_bbox():
    assert roi_bbox(ROI) == BBox(x=40, y=60, w=240, h=160)
