"""Frame-differencing motion detector. Pure numpy/OpenCV; no camera here."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

Point = tuple[int, int]


class BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


MIN_BLOB_AREA = 4  # changed (pre-dilation) pixels; smaller blobs are noise specks
_JOIN_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class MotionResult:
    score: int
    triggered: bool
    # Union box of the changed-pixel blobs inside the ROI, lores px; set only on a trigger.
    bbox: BBox | None = None


def roi_bbox(roi: Sequence[Point]) -> BBox:
    """Axis-aligned extent of the ROI polygon.

    `w`/`h` are max-min of the vertex coordinates (the polygon's extent).
    Note that the rasterised mask used by `MotionDetector` includes the
    boundary pixel, so it covers (w + 1) x (h + 1) pixels.
    """
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    return BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


def blob_bbox(changed: np.ndarray, min_area: int = MIN_BLOB_AREA) -> BBox | None:
    """Union box of the blobs in a 0/1 changed-pixel mask.

    A 3x3 dilation joins a bird's parts (head, tail, wing) into one blob; blobs with fewer
    than `min_area` changed pixels *before* dilation are dropped as noise.
    """
    joined = cv2.dilate(changed, _JOIN_KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return None
    original = np.bincount(labels[changed > 0], minlength=n)
    keep = [i for i in range(1, n) if original[i] >= min_area]
    if not keep:
        return None
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


class MotionDetector:
    """Compares each greyscale frame to a running-average background.

    `process()` must be called with frames of shape (h, w) uint8 where
    (w, h) == frame_size. The ROI polygon is in the same coordinates.

    Warm-up: the first `warmup_frames` processed frames (counting the
    background-initialising first frame) never trigger, although the
    background is still updated and the score is still reported. This
    absorbs the camera's auto-exposure settling after start-up, which
    otherwise shifts the whole ROI and would fire a few junk captures.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        roi: Sequence[Point],
        pixel_threshold: int,
        min_area: int,
        background_alpha: float,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        warmup_frames: int = 10,
    ) -> None:
        w, h = frame_size
        self._mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [np.array(roi, dtype=np.int32)], 1)
        self._pixel_threshold = pixel_threshold
        self._min_area = min_area
        self._alpha = background_alpha
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._warmup_frames = warmup_frames
        self._frames_seen = 0
        self._bg: np.ndarray | None = None
        self._last_trigger = float("-inf")

    def process(self, frame: np.ndarray) -> MotionResult:
        self._frames_seen += 1
        blurred = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
        if self._bg is None:
            self._bg = blurred
            return MotionResult(score=0, triggered=False)

        diff = np.abs(blurred - self._bg)
        changed = (diff > self._pixel_threshold).astype(np.uint8) * self._mask
        score = int(changed.sum())

        self._bg = (1.0 - self._alpha) * self._bg + self._alpha * blurred

        now = self._clock()
        warming_up = self._frames_seen <= self._warmup_frames
        triggered = (
            not warming_up
            and score >= self._min_area
            and (now - self._last_trigger) >= self._cooldown
        )
        if triggered:
            self._last_trigger = now
        # Only a trigger consumes the box: skip the blob search on every other frame.
        bbox = blob_bbox(changed) if triggered else None
        return MotionResult(score=score, triggered=triggered, bbox=bbox)


MIN_CROP_SIDE = 40         # lores px (~410 px of a 3280-wide still)
MAX_MOTION_FRACTION = 0.5  # motion bigger than this share of the ROI box = lighting/feeder, not a bird
ROI_PAD = 0.1              # the ROI fallback crop's padding (today's behaviour)

NormBox = tuple[float, float, float, float]  # x, y, w, h as fractions of the lores frame


@dataclass(frozen=True)
class CropBox:
    bbox: BBox   # lores px
    pad: float   # fraction added on each side by `crop_to_bbox`
    kind: str    # 'motion' | 'roi'


def normalise_bbox(bbox: BBox, frame_size: tuple[int, int]) -> NormBox:
    fw, fh = frame_size
    return (bbox.x / fw, bbox.y / fh, bbox.w / fw, bbox.h / fh)


def denormalise_bbox(box: NormBox, frame_size: tuple[int, int]) -> BBox:
    fw, fh = frame_size
    x, y, w, h = box
    return BBox(x=round(x * fw), y=round(y * fh), w=round(w * fw), h=round(h * fh))


def motion_crop(bbox: BBox | None, roi: Sequence[Point], frame_size: tuple[int, int]) -> BBox | None:
    """Square crop around the motion box, or None to fall back to the ROI crop.

    Side = 2x the box's longer side (the still is taken a few hundred ms after the trigger
    frame, so the bird may have moved), at least MIN_CROP_SIDE, at most the frame's short
    side; shifted, never shrunk, to stay inside the frame. Square so the model's 224x224
    resize doesn't distort the bird.
    """
    if bbox is None:
        return None
    rb = roi_bbox(roi)
    if bbox.w * bbox.h > MAX_MOTION_FRACTION * rb.w * rb.h:
        return None
    fw, fh = frame_size
    side = min(max(2 * max(bbox.w, bbox.h), MIN_CROP_SIDE), fw, fh)
    x = round(bbox.x + bbox.w / 2 - side / 2)
    y = round(bbox.y + bbox.h / 2 - side / 2)
    x = min(max(x, 0), fw - side)
    y = min(max(y, 0), fh - side)
    return BBox(x=x, y=y, w=side, h=side)


def crop_box(motion_box: NormBox | None, roi: Sequence[Point], frame_size: tuple[int, int]) -> CropBox:
    """The crop to classify for a capture: around its stored motion box, else today's ROI crop."""
    bbox = None if motion_box is None else denormalise_bbox(motion_box, frame_size)
    square = motion_crop(bbox, roi, frame_size)
    if square is not None:
        return CropBox(bbox=square, pad=0.0, kind="motion")
    return CropBox(bbox=roi_bbox(roi), pad=ROI_PAD, kind="roi")
