"""Bird / not-bird classification with a TFLite ImageNet model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from birdy.motion import BBox

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    confidence: float  # P(bird), 0..1
    species: str | None = None
    species_confidence: float | None = None


class Classifier(Protocol):
    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction: ...


def _load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older Pi installs
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=str(model_path))


def load_bird_indices(path: Path) -> frozenset[int]:
    out: set[int] = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "-" in line:
            a, b = line.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(line))
    return frozenset(out)


def bird_probability(probs: np.ndarray, bird_indices: frozenset[int]) -> float:
    return float(sum(probs[i] for i in bird_indices if i < len(probs)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def crop_to_bbox(img: np.ndarray, bbox: BBox, lores_size: tuple[int, int], pad: float) -> np.ndarray:
    """Crop `img` (main-stream size) to `bbox` given in lores coordinates, padded by `pad` fraction."""
    img_h, img_w = img.shape[:2]
    sx = img_w / lores_size[0]
    sy = img_h / lores_size[1]
    x0 = (bbox.x - bbox.w * pad) * sx
    y0 = (bbox.y - bbox.h * pad) * sy
    x1 = (bbox.x + bbox.w * (1 + pad)) * sx
    y1 = (bbox.y + bbox.h * (1 + pad)) * sy
    # Clamp both ends into [0, size] so a bbox outside the frame yields an empty
    # crop rather than a negative index that wraps the slice.
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1 = max(0, min(img_w, int(round(x1))))
    y1 = max(0, min(img_h, int(round(y1))))
    return img[y0:y1, x0:x1]


class ImageNetBirdClassifier:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        bird_classes_path: Path,
        threshold: float,
        lores_size: tuple[int, int],
        pad: float = 0.1,
    ) -> None:
        self._interp = _load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self._labels = Path(labels_path).read_text().splitlines() if Path(labels_path).exists() else []
        self._bird_indices = load_bird_indices(bird_classes_path)
        self._threshold = threshold
        self._lores_size = lores_size
        self._pad = pad
        self._warned_bbox: BBox | None = None

    def _probs(self, image_path: Path, roi_bbox: BBox | None) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        if roi_bbox is not None:
            crop = crop_to_bbox(img, roi_bbox, self._lores_size, self._pad)
            if crop.size == 0:
                # A bad ROI must not turn every capture into an error, so fall
                # back to the full frame. Warn once per distinct bbox: the ROI is
                # live-tunable, so a new bad one deserves its own warning.
                if roi_bbox != self._warned_bbox:
                    log.warning("ROI %s yields an empty crop; using the full frame", roi_bbox)
                    self._warned_bbox = roi_bbox
            else:
                img = crop
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)

        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 127.5) - 1.0
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        out = self._interp.get_tensor(self._output["index"])[0]

        scale, zero_point = self._output["quantization"]
        if scale:
            out = (out.astype(np.float32) - zero_point) * scale
        else:
            out = out.astype(np.float32)
        # The hosted mobilenet_v2_1.0_224_quant model has no softmax layer: its
        # output dequantizes to logits (scale ~0.099, zero point 58).
        return _softmax(out)

    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction:
        """`threshold` overrides the constructor default so it can be tuned live."""
        t = self._threshold if threshold is None else threshold
        conf = bird_probability(self._probs(image_path, roi_bbox), self._bird_indices)
        return Prediction(is_bird=conf >= t, confidence=conf)

    def top_k(self, image_path: Path, roi_bbox: BBox | None, k: int = 5) -> list[tuple[str, float]]:
        probs = self._probs(image_path, roi_bbox)
        order = np.argsort(probs)[::-1][:k]
        return [(self._labels[i] if i < len(self._labels) else str(i), float(probs[i])) for i in order]
