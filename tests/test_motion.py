import numpy as np

from birdy.motion import BBox, MotionDetector, blob_bbox, roi_bbox

W, H = 320, 240
ROI = ((40, 60), (280, 60), (280, 220), (40, 220))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_detector(clock, **overrides):
    kw = dict(frame_size=(W, H), roi=ROI, pixel_threshold=25, min_area=150, background_alpha=0.05, cooldown_seconds=3.0, clock=clock, warmup_frames=0)
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
    # changed region is ~32x32 minus corners (measured 1020).
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
    triggers = 0
    for i in range(100):
        clock.t = i * 0.1  # cooldown (3 s) is active for most of the run
        r = d.process(with_blob(100, 100))
        triggers += r.triggered
    assert r.score == 0
    # Background keeps updating during cooldown, so the blob fades after at
    # most one re-trigger instead of firing on every cooldown expiry.
    assert triggers <= 2


def test_small_blob_below_min_area_does_not_trigger():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100, size=5))
    assert 0 < r.score < 150 and r.triggered is False


def test_warmup_suppresses_triggers_but_updates_background():
    clock = FakeClock()
    d = make_detector(clock, warmup_frames=10)
    d.process(blank())
    brighter = np.full((H, W), 80, dtype=np.uint8)  # +30 exposure shift
    results = []
    for i in range(12):
        clock.t = (i + 1) * 10.0
        results.append(d.process(brighter))
    # The whole ROI changes on the first bright frame, but warm-up blocks it.
    assert results[0].score > 10_000
    assert not any(r.triggered for r in results[:10])
    # Background absorbed the shift even though nothing triggered.
    assert results[11].score == 0
    # Once warmed up, real motion still fires.
    clock.t = 200.0
    assert d.process(with_blob(100, 100)).triggered is True


def test_non_rectangular_roi_uses_polygon_not_bbox():
    tri = ((40, 60), (280, 60), (160, 220))
    clock = FakeClock()
    d = make_detector(clock, roi=tri)
    d.process(blank())
    # Inside the bounding box, outside the triangle.
    r = d.process(with_blob(60, 180))
    assert r.score == 0 and r.triggered is False
    # Inside the triangle.
    r = d.process(with_blob(145, 100))
    assert r.triggered is True


def test_roi_bbox():
    assert roi_bbox(ROI) == BBox(x=40, y=60, w=240, h=160)


def test_trigger_reports_blob_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100))
    assert r.bbox is not None
    # 30x30 blob, plus the blur halo and the 3x3 dilation: a few px larger on each side.
    assert 95 <= r.bbox.x <= 100 and 95 <= r.bbox.y <= 100
    assert 30 <= r.bbox.w <= 36 and 30 <= r.bbox.h <= 36


def test_no_motion_has_no_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    assert d.process(blank()).bbox is None


def test_blob_outside_roi_has_no_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    assert d.process(with_blob(0, 0)).bbox is None


def test_blob_bbox_unions_blobs_and_drops_specks():
    changed = np.zeros((H, W), np.uint8)
    changed[100:110, 100:110] = 1     # blob A
    changed[150:160, 200:210] = 1     # blob B
    changed[50, 50] = 1               # 1 px speck: dropped (< MIN_BLOB_AREA original pixels)
    b = blob_bbox(changed)
    assert b is not None
    assert b.x <= 100 and b.y <= 100
    assert b.x + b.w >= 210 and b.y + b.h >= 160
    assert b.x >= 98 and b.y >= 98    # the speck at (50, 50) did not widen it


def test_blob_bbox_none_for_specks_only():
    changed = np.zeros((H, W), np.uint8)
    changed[50, 50] = 1
    changed[200, 300] = 1
    assert blob_bbox(changed) is None
