from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.detector import BirdDetector, Detection

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "efficientdet_lite0_320_ptq.tflite"
LABELS = ROOT / "models" / "coco_labels.txt"
FIXTURES = Path(__file__).parent / "fixtures"
IMG = np.zeros((100, 120, 3), np.uint8)


class FakeInterpreter:
    """Post-processed detector outputs in a configurable tensor order."""

    def __init__(self, order, dtype=np.uint8):
        self.order = order  # names in output order, from {"boxes", "classes", "scores", "count"}
        self.dtype = dtype
        self.tensors = {
            "boxes": np.array([[[0.1, 0.2, 0.5, 0.6], [0.0, 0.0, 1.5, 1.5], [0, 0, 0, 0]]], np.float32),
            "classes": np.array([[15.0, 15.0, 0.0]], np.float32),
            "scores": np.array([[0.62, 0.31, 0.0]], np.float32),
            "count": np.array([2.0], np.float32),
        }
        self.fed = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 320, 320, 3]), "dtype": self.dtype, "quantization": (0.0078125, 127)}]

    def get_output_details(self):
        return [{"index": i + 1, "name": n} for i, n in enumerate(self.order)]

    def set_tensor(self, index, value):
        self.fed = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.tensors[self.order[index - 1]]


def fake_detector(monkeypatch, tmp_path, order, dtype=np.uint8, labels="person\nbicycle\n" + "???\n" * 13 + "bird\n"):
    import birdy.detector as det

    interp = FakeInterpreter(order, dtype)
    monkeypatch.setattr(det, "load_interpreter", lambda path: interp)
    lab = tmp_path / "labels.txt"
    lab.write_text(labels)
    return BirdDetector(tmp_path / "model.tflite", lab), interp


def test_efficientdet_order_decodes(monkeypatch, tmp_path):
    d, interp = fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"])
    dets = d.detect(IMG)
    assert dets == [
        Detection("bird", pytest.approx(0.62), (pytest.approx(0.2), pytest.approx(0.1), pytest.approx(0.4), pytest.approx(0.4))),
        Detection("bird", pytest.approx(0.31), (0.0, 0.0, 1.0, 1.0)),   # clamped
    ]
    assert interp.fed.dtype == np.uint8 and interp.fed.shape == (1, 320, 320, 3)


def test_ssd_order_decodes_the_same(monkeypatch, tmp_path):
    a, _ = fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"])
    b, _ = fake_detector(monkeypatch, tmp_path, ["count", "scores", "classes", "boxes"])
    assert a.detect(IMG) == b.detect(IMG)


def test_best_bird_is_the_top_bird(monkeypatch, tmp_path):
    d, _ = fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"])
    assert d.best_bird(IMG).score == pytest.approx(0.62)


def test_best_bird_none_without_bird(monkeypatch, tmp_path):
    d, interp = fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"])
    interp.tensors["classes"] = np.array([[0.0, 0.0, 0.0]], np.float32)   # person
    assert d.best_bird(IMG) is None
    assert d.detect(IMG)[0].label == "person"


def test_float_model_gets_scaled_input(monkeypatch, tmp_path):
    d, interp = fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"], dtype=np.float32)
    d.detect(np.full((50, 50, 3), 255, np.uint8))
    assert interp.fed.dtype == np.float32 and float(interp.fed.max()) == pytest.approx(1.0)


def test_missing_bird_label_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="bird"):
        fake_detector(monkeypatch, tmp_path, ["boxes", "classes", "scores", "count"], labels="person\ncat\n")


@pytest.fixture(scope="module")
def det():
    return BirdDetector(MODEL, LABELS)


@pytest.mark.skipif(not MODEL.exists() or not (FIXTURES / "bird.jpg").exists(), reason="model or fixture missing")
class TestRealModel:
    def test_bird_fixture_has_a_bird(self, det):
        img = cv2.cvtColor(cv2.imread(str(FIXTURES / "bird.jpg")), cv2.COLOR_BGR2RGB)
        b = det.best_bird(img)
        assert b is not None and b.score >= 0.3
        assert all(0.0 <= v <= 1.0 for v in b.box)

    def test_cat_fixture_has_no_confident_bird(self, det):
        img = cv2.cvtColor(cv2.imread(str(FIXTURES / "cat.jpg")), cv2.COLOR_BGR2RGB)
        b = det.best_bird(img)
        assert b is None or b.score < 0.2
