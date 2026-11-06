"""Three models on one crop: an ImageNet bird/not-bird gate, a COCO bird detector and a species model. Gate or detector makes a bird; a frame both rejected is not a bird and never reaches the species model."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from birdy.detector import BirdDetector
from birdy.live import LiveConfig
from birdy.motion import BBox
from birdy.species import InatBirdClassifier, SpeciesLabel
from birdy.tflite import load_interpreter

if TYPE_CHECKING:
    from birdy.config import Config
    from birdy.species_list import SpeciesList

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    # How sure the system is this is a bird: max(gate P(bird), detector score); 0..1.
    confidence: float
    species: str | None = None
    species_confidence: float | None = None
    detector_confidence: float = 0.0   # best COCO "bird" detection score; 0 when none
    detector_box: tuple[float, float, float, float] | None = None   # (x, y, w, h) fractions of the crop


class Classifier(Protocol):
    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction: ...


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


class CropLoader:
    """Decode a still and crop it to the ROI bbox (lores coords -> main coords), as RGB."""

    def __init__(self, lores_size: tuple[int, int], pad: float = 0.1) -> None:
        self._lores_size = lores_size
        self._pad = pad
        self._warned_bbox: BBox | None = None

    def load(self, image_path: Path, roi_bbox: BBox | None, pad: float | None = None) -> np.ndarray:
        """`pad` overrides the constructor's padding (a motion crop is already padded: 0)."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        if roi_bbox is not None:
            crop = crop_to_bbox(img, roi_bbox, self._lores_size, self._pad if pad is None else pad)
            if crop.size == 0:
                # A bad ROI must not turn every capture into an error, so fall back to the
                # full frame. Warn once per distinct bbox: the ROI is live-tunable.
                if roi_bbox != self._warned_bbox:
                    log.warning("ROI %s yields an empty crop; using the full frame", roi_bbox)
                    self._warned_bbox = roi_bbox
            else:
                img = crop
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
        self._interp = load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self._labels = Path(labels_path).read_text().splitlines() if Path(labels_path).exists() else []
        self._bird_indices = load_bird_indices(bird_classes_path)
        self._threshold = threshold
        self._crops = CropLoader(lores_size, pad)

    def _probs(self, img_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)

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

    def predict(self, img_rgb: np.ndarray) -> float:
        """P(bird) for an already-decoded RGB image."""
        return bird_probability(self._probs(img_rgb), self._bird_indices)

    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction:
        """`threshold` overrides the constructor default so it can be tuned live."""
        t = self._threshold if threshold is None else threshold
        conf = self.predict(self._crops.load(image_path, roi_bbox, pad))
        return Prediction(is_bird=conf >= t, confidence=conf)

    def top_k(self, image_path: Path, roi_bbox: BBox | None, k: int = 5) -> list[tuple[str, float]]:
        return self.top_k_image(self._crops.load(image_path, roi_bbox), k)

    def top_k_image(self, img_rgb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        probs = self._probs(img_rgb)
        order = np.argsort(probs)[::-1][:k]
        return [(self._labels[i] if i < len(self._labels) else str(i), float(probs[i])) for i in order]


class TwoStageClassifier:
    """Three signals on the same decoded crop: the ImageNet gate (bird / not-bird), an optional
    COCO detector and the species model. Implements `Classifier`.

    Rules (thresholds from `live`): the gate's P(bird) at or above `classifier.threshold`
    makes the photo a bird. A COCO `bird` detection at or above `detector.threshold` makes
    the photo a bird too. A frame both gate and detector rejected is not a bird, and the
    species model is not run on it at all (it has no "nothing here" answer). For a bird, a
    species is reported when its probability meets `species.threshold`.
    With a `species_list`, the species model's candidate set is re-applied (before a species
    run, under the lock) whenever the `[species]` config or the list's version changed.
    A failing detector, species model or candidate refresh is logged once per distinct error
    and skipped (a bird then has no species; a failed refresh is retried on the next bird).
    """

    def __init__(
        self,
        gate: ImageNetBirdClassifier,
        species: InatBirdClassifier,
        live: LiveConfig,
        lores_size: tuple[int, int],
        pad: float = 0.1,
        detector: BirdDetector | None = None,
        species_list: SpeciesList | None = None,
    ) -> None:
        self.gate = gate
        self.species = species
        self.detector = detector
        self.species_list = species_list
        self._cand_key: tuple | None = None
        self._live = live
        self._crops = CropLoader(lores_size, pad)
        self._species_errors: set[str] = set()
        self._detector_errors: set[str] = set()
        # TFLite interpreters are not thread-safe; the pipeline and the web thread share this object.
        self._lock = threading.Lock()

    def classify_image(
        self, img_rgb: np.ndarray, threshold: float | None = None
    ) -> tuple[Prediction, list[tuple[SpeciesLabel, float]]]:
        cfg = self._live.get()
        t_gate = cfg.classifier.threshold if threshold is None else threshold
        t_det = cfg.detector.threshold
        t_sp = cfg.species.threshold

        with self._lock:
            p_bird = self.gate.predict(img_rgb)
            det = None
            if self.detector is not None:
                try:
                    det = self.detector.best_bird(img_rgb)
                except Exception as e:
                    key = f"{type(e).__name__}: {e}"
                    if key not in self._detector_errors:  # log each distinct failure once, not per capture
                        log.exception("detector failed (%s); continuing without it", key)
                        self._detector_errors.add(key)
            p_det = det.score if det is not None else 0.0
            is_bird = p_bird >= t_gate or p_det >= t_det
            top = []
            if is_bird:
                # The species model has no "nothing here" answer (14 % of empty frames get a confident
                # name), so it only ever sees photos the gate or detector already called a bird.
                try:
                    self._refresh_candidates(cfg)
                    top = self.species.top(img_rgb, k=5)
                except Exception as e:
                    key = f"{type(e).__name__}: {e}"
                    if key not in self._species_errors:
                        log.exception("species model failed (%s); continuing without a species", key)
                        self._species_errors.add(key)
                    top = []
        label, p_sp = top[0] if top else (None, 0.0)
        species = label.common if label is not None and p_sp >= t_sp else None
        pred = Prediction(
            is_bird=is_bird,
            confidence=max(p_bird, p_det),
            species=species,
            species_confidence=p_sp if species is not None else None,
            detector_confidence=p_det,
            detector_box=det.box if det is not None else None,
        )
        return pred, top

    def _refresh_candidates(self, cfg: Config) -> None:
        """Re-apply the candidate set when the `[species]` config or the local list changed. Caller holds the lock."""
        if self.species_list is None:
            return
        # version is read before candidates(): a reload in between gives one redundant recompute, never a stale set.
        key = (self.species_list.version, cfg.species)
        if key == self._cand_key:
            return
        n = self.species.set_candidates(self.species_list.candidates(cfg.species))
        log.info("species candidates: %d", n)
        self._cand_key = key

    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction:
        return self.classify_image(self._crops.load(image_path, roi_bbox, pad), threshold)[0]
