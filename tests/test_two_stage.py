import dataclasses
import logging

import numpy as np
import pytest

from birdy.classifier import CropLoader, TwoStageClassifier
from birdy.species import SpeciesLabel

BLUE_TIT = SpeciesLabel("Cyanistes caeruleus", "Eurasian Blue Tit")
IMG = np.zeros((100, 100, 3), np.uint8)


class FakeGate:
    def __init__(self, p):
        self.p = p

    def predict(self, img):
        return self.p


class FakeSpecies:
    def __init__(self, p, raise_exc=None):
        self.p = p
        self.raise_exc = raise_exc
        self.calls = 0

    def top(self, img, k=5, skip_background=True):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return [(BLUE_TIT, self.p), (SpeciesLabel("Parus major", "Great Tit"), self.p / 2)][:k]


def two_stage(live, p_bird, p_sp, raise_exc=None):
    return TwoStageClassifier(FakeGate(p_bird), FakeSpecies(p_sp, raise_exc), live, lores_size=(320, 240))


def test_bird_with_species(live):
    pred, top = two_stage(live, 0.9, 0.8).classify_image(IMG)
    assert pred.is_bird and pred.confidence == 0.9
    assert pred.species == "Eurasian Blue Tit" and pred.species_confidence == 0.8
    assert top[0][0] == BLUE_TIT


def test_bird_with_unknown_species(live):
    pred, _ = two_stage(live, 0.9, 0.3).classify_image(IMG)
    assert pred.is_bird and pred.species is None and pred.species_confidence is None


def test_rescue_above_rescue_threshold(live):
    pred, _ = two_stage(live, 0.2, 0.75).classify_image(IMG)
    assert pred.is_bird is True
    assert pred.confidence == 0.75  # max(p_bird, p_species) when rescued
    assert pred.species == "Eurasian Blue Tit"


def test_no_rescue_between_thresholds(live):
    pred, _ = two_stage(live, 0.2, 0.6).classify_image(IMG)
    assert pred.is_bird is False and pred.confidence == 0.2
    assert pred.species == "Eurasian Blue Tit"  # label is still worth showing on a non-bird tile


def test_thresholds_follow_live_config(live, config):
    clf = two_stage(live, 0.2, 0.6)
    assert clf.classify_image(IMG)[0].is_bird is False
    live.replace(dataclasses.replace(config, species=dataclasses.replace(config.species, threshold=0.5, rescue_threshold=0.55)))
    assert clf.classify_image(IMG)[0].is_bird is True


def test_gate_threshold_override(live):
    clf = two_stage(live, 0.45, 0.1)
    assert clf.classify_image(IMG)[0].is_bird is False
    assert clf.classify_image(IMG, threshold=0.4)[0].is_bird is True


def test_species_failure_keeps_gate_result(live, caplog):
    clf = two_stage(live, 0.9, 0.0, raise_exc=RuntimeError("boom"))
    with caplog.at_level(logging.ERROR):
        pred, top = clf.classify_image(IMG)
        clf.classify_image(IMG)
    assert pred.is_bird and pred.species is None and top == []
    assert caplog.text.count("species model failed") == 1  # deduplicated


def test_classify_path_uses_crop_loader(live, tmp_path):
    import cv2

    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), np.full((240, 320, 3), 90, np.uint8))
    pred = two_stage(live, 0.9, 0.8).classify(p, roi_bbox=None)
    assert pred.is_bird and pred.species == "Eurasian Blue Tit"


def test_crop_loader_rgb_and_fallback(tmp_path, caplog):
    import cv2

    from birdy.motion import BBox

    p = tmp_path / "x.png"  # lossless, so the exact channel values survive
    bgr = np.zeros((240, 320, 3), np.uint8)
    bgr[:, :, 0] = 255  # blue in BGR
    cv2.imwrite(str(p), bgr)
    loader = CropLoader((320, 240))
    img = loader.load(p, None)
    assert img[0, 0].tolist() == [0, 0, 255]  # RGB now
    with caplog.at_level(logging.WARNING):
        full = loader.load(p, BBox(1000, 1000, 10, 10))
        loader.load(p, BBox(1000, 1000, 10, 10))
    assert full.shape == (240, 320, 3)
    assert caplog.text.count("empty crop") == 1
    with pytest.raises(ValueError):
        loader.load(tmp_path / "missing.jpg", None)


def test_classify_image_is_serialised_across_threads(live):
    import threading
    import time

    class BusySpecies(FakeSpecies):
        def __init__(self):
            super().__init__(0.8)
            self.busy = False
            self.overlaps = 0

        def top(self, img, k=5, skip_background=True):
            if self.busy:
                self.overlaps += 1
            self.busy = True
            time.sleep(0.001)
            self.busy = False
            return super().top(img, k, skip_background)

    species = BusySpecies()
    clf = TwoStageClassifier(FakeGate(0.9), species, live, lores_size=(320, 240))
    threads = [threading.Thread(target=lambda: [clf.classify_image(IMG) for _ in range(50)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert species.calls == 200 and species.overlaps == 0
