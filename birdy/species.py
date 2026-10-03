"""Bird species recognition with the Coral iNaturalist TFLite model (965 classes)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from birdy.tflite import load_interpreter

_LABEL_RE = re.compile(r"^(.*?)\s*\((.*)\)\s*$")


@dataclass(frozen=True)
class SpeciesLabel:
    latin: str
    common: str


def parse_label(line: str) -> SpeciesLabel:
    """`Cyanistes caeruleus (Eurasian Blue Tit)` -> latin/common; a bare name is used for both."""
    line = line.strip()
    m = _LABEL_RE.match(line)
    if m:
        return SpeciesLabel(m.group(1).strip(), m.group(2).strip())
    return SpeciesLabel(line, line)


class InatBirdClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self._interp = load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self.labels = [parse_label(l) for l in Path(labels_path).read_text().splitlines() if l.strip()]
        n_out = int(self._output["shape"][-1])
        if len(self.labels) != n_out:
            raise ValueError(f"{labels_path}: {len(self.labels)} labels but the model has {n_out} outputs")

    def predict(self, img_rgb: np.ndarray) -> list[tuple[SpeciesLabel, float]]:
        """All classes with their probability, best first. The model's output is already
        softmax-scaled (uint8, scale 1/256), so it is only dequantised — never re-softmaxed."""
        img = cv2.resize(img_rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)
        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 127.5) - 1.0
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        out = self._interp.get_tensor(self._output["index"])[0]
        scale, zero_point = self._output["quantization"]
        probs = (out.astype(np.float32) - zero_point) * scale if scale else out.astype(np.float32)
        order = np.argsort(probs)[::-1]
        return [(self.labels[i], float(probs[i])) for i in order]

    def top(self, img_rgb: np.ndarray, k: int = 5, skip_background: bool = True) -> list[tuple[SpeciesLabel, float]]:
        ranked = self.predict(img_rgb)
        if skip_background:
            ranked = [(l, p) for l, p in ranked if l.latin != "background"]
        return ranked[:k]
