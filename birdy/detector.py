"""COCO object detection (Coral EfficientDet-Lite0 / SSD-MobileNet TFLite) used as a second
bird/not-bird signal. The model's own post-processing gives boxes, classes, scores and a count."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from birdy.tflite import load_interpreter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # (x, y, w, h) as fractions of the input image, clamped to [0, 1]


class BirdDetector:
    """Not thread-safe (TFLite); callers serialise through `TwoStageClassifier._lock`."""

    def __init__(self, model_path: Path, labels_path: Path, bird_label: str = "bird") -> None:
        # One name per line; line index = class id ("n/a" marks unused COCO ids).
        self.labels = [l.strip() for l in Path(labels_path).read_text().splitlines()]
        if bird_label not in self.labels:
            raise ValueError(f"{labels_path}: no '{bird_label}' label")
        self.bird_label = bird_label
        self._interp = load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self._outputs = self._interp.get_output_details()

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        """Every detection the model returned, best first."""
        img = cv2.resize(img_rgb, (int(self._in_w), int(self._in_h)), interpolation=cv2.INTER_AREA)
        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 255.0)
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        tensors = [np.asarray(self._interp.get_tensor(d["index"])) for d in self._outputs]
        # Match outputs by shape, not position: Coral's SSD and EfficientDet order them differently.
        boxes = next(t[0] for t in tensors if t.ndim == 3)
        count = int(next(t for t in tensors if t.ndim == 1)[0])
        vecs = [t[0] for t in tensors if t.ndim == 2]
        a, b = vecs[0], vecs[1]
        if _integral(a) and not _integral(b):
            classes, scores = a, b
        elif _integral(b) and not _integral(a):
            classes, scores = b, a
        else:  # both integral (all zeros) or neither: scores are the ones within [0, 1]
            classes, scores = (a, b) if float(a.max(initial=0)) > 1.0 else (b, a)
        out = []
        for i in range(min(count, len(scores), len(classes), len(boxes))):
            cid = int(round(float(classes[i])))
            label = self.labels[cid] if 0 <= cid < len(self.labels) else str(cid)
            y0, x0, y1, x1 = (min(1.0, max(0.0, float(v))) for v in boxes[i])
            out.append(Detection(label, float(scores[i]), (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))))
        out.sort(key=lambda d: -d.score)
        return out

    def best_bird(self, img_rgb: np.ndarray) -> Detection | None:
        birds = [d for d in self.detect(img_rgb) if d.label == self.bird_label]
        return birds[0] if birds else None


def _integral(v: np.ndarray) -> bool:
    return bool(np.all(np.abs(v - np.round(v)) < 1e-6))
