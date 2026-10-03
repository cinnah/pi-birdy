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


@dataclass(frozen=True)
class MotionResult:
    score: int
    triggered: bool


def roi_bbox(roi: Sequence[Point]) -> BBox:
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    return BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


class MotionDetector:
    """Compares each greyscale frame to a running-average background.

    `process()` must be called with frames of shape (h, w) uint8 where
    (w, h) == frame_size. The ROI polygon is in the same coordinates.
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
    ) -> None:
        w, h = frame_size
        self._mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [np.array(roi, dtype=np.int32)], 1)
        self._pixel_threshold = pixel_threshold
        self._min_area = min_area
        self._alpha = background_alpha
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._bg: np.ndarray | None = None
        self._last_trigger = float("-inf")

    def process(self, frame: np.ndarray) -> MotionResult:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
        if self._bg is None:
            self._bg = blurred
            return MotionResult(score=0, triggered=False)

        diff = np.abs(blurred - self._bg)
        changed = (diff > self._pixel_threshold).astype(np.uint8) * self._mask
        score = int(changed.sum())

        self._bg = (1.0 - self._alpha) * self._bg + self._alpha * blurred

        now = self._clock()
        triggered = score >= self._min_area and (now - self._last_trigger) >= self._cooldown
        if triggered:
            self._last_trigger = now
        return MotionResult(score=score, triggered=triggered)
