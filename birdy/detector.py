"""COCO object detection (Coral EfficientDet-Lite0 / SSD-MobileNet TFLite) used as a second
bird/not-bird signal. The model's own post-processing gives boxes, classes, scores and a count."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from birdy.tflite import load_interpreter


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
        outputs = self._interp.get_output_details()
        shapes = [tuple(int(n) for n in d["shape"]) for d in outputs]
        by_ndim = {k: [d["index"] for d in outputs if len(d["shape"]) == k] for k in (1, 2, 3)}
        if len(outputs) != 4 or len(by_ndim[3]) != 1 or len(by_ndim[1]) != 1 or len(by_ndim[2]) != 2:
            raise ValueError(f"{model_path}: unexpected detector outputs {shapes}")
        # Matched by shape, not position: Coral's SSD and EfficientDet order them differently.
        self._boxes_idx = by_ndim[3][0]
        self._count_idx = by_ndim[1][0]
        self._vec_idx = tuple(by_ndim[2])

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        """Every detection the model returned, best first."""
        img = cv2.resize(img_rgb, (int(self._in_w), int(self._in_h)), interpolation=cv2.INTER_AREA)
        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 255.0)
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        boxes = np.asarray(self._interp.get_tensor(self._boxes_idx))[0]
        count = int(np.asarray(self._interp.get_tensor(self._count_idx))[0])
        a, b = (np.asarray(self._interp.get_tensor(i))[0] for i in self._vec_idx)
        if _integral(a) and not _integral(b):
            classes, scores = a, b
        elif _integral(b) and not _integral(a):
            classes, scores = b, a
        else:  # both integral (all zeros) or neither: scores are the ones within [0, 1]
            # (both integral with max <= 1 only happens with count = 0, where the loop below is empty)
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
