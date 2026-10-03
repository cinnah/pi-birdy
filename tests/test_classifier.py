from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import ImageNetBirdClassifier, Prediction, bird_probability, crop_to_bbox, load_bird_indices
from birdy.motion import BBox

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite"
LABELS = ROOT / "models" / "imagenet_labels.txt"
BIRD_CLASSES = ROOT / "models" / "imagenet_bird_classes.txt"
FIXTURES = Path(__file__).parent / "fixtures"


def test_load_bird_indices_parses_ranges_and_comments(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("# comment\n3\n\n10-12\n")
    assert load_bird_indices(p) == frozenset({3, 10, 11, 12})


def test_load_bird_indices_real_file():
    idx = load_bird_indices(BIRD_CLASSES)
    assert 8 in idx and 25 in idx and 147 in idx
    assert 7 not in idx and 26 not in idx
    assert len(idx) == 18 + 21 + 20


def test_bird_probability_sums_bird_indices():
    probs = np.zeros(10, dtype=np.float32)
    probs[2] = 0.3
    probs[5] = 0.4
    probs[9] = 0.3
    assert bird_probability(probs, frozenset({2, 5})) == pytest.approx(0.7)


def test_crop_to_bbox_scales_from_lores_to_main_and_pads():
    img = np.zeros((2464, 3280, 3), dtype=np.uint8)
    # bbox in 320x240 coords; scale is 10.25 x ~10.27
    out = crop_to_bbox(img, BBox(x=40, y=60, w=240, h=160), lores_size=(320, 240), pad=0.1)
    h, w = out.shape[:2]
    assert 2800 <= w <= 3000 and 1900 <= h <= 2100


def test_crop_to_bbox_clamps_to_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out = crop_to_bbox(img, BBox(x=0, y=0, w=320, h=240), lores_size=(320, 240), pad=0.5)
    assert out.shape[:2] == (240, 320)


@pytest.fixture(scope="module")
def clf():
    return ImageNetBirdClassifier(MODEL, LABELS, BIRD_CLASSES, threshold=0.5, lores_size=(320, 240))


@pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded")
class TestRealModel:
    @pytest.mark.skipif(not (FIXTURES / "bird.jpg").exists(), reason="fixture missing")
    def test_bird_is_bird(self, clf):
        p = clf.classify(FIXTURES / "bird.jpg", roi_bbox=None)
        assert isinstance(p, Prediction)
        assert p.is_bird and p.confidence > 0.5

    @pytest.mark.skipif(not (FIXTURES / "cat.jpg").exists(), reason="fixture missing")
    def test_cat_is_not_bird(self, clf):
        p = clf.classify(FIXTURES / "cat.jpg", roi_bbox=None)
        assert not p.is_bird and p.confidence < 0.2

    @pytest.mark.skipif(not (FIXTURES / "bird.jpg").exists(), reason="fixture missing")
    def test_top_k_returns_labels(self, clf):
        top = clf.top_k(FIXTURES / "bird.jpg", roi_bbox=None, k=3)
        assert len(top) == 3
        assert all(isinstance(label, str) and 0 <= prob <= 1 for label, prob in top)

    def test_grey_image_is_not_bird(self, clf, tmp_path):
        p = tmp_path / "grey.jpg"
        cv2.imwrite(str(p), np.full((480, 640, 3), 128, dtype=np.uint8))
        assert clf.classify(p, roi_bbox=None).is_bird is False

    def test_unreadable_image_raises(self, clf, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not a jpeg")
        with pytest.raises(ValueError):
            clf.classify(p, roi_bbox=None)
