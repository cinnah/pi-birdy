from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.species import InatBirdClassifier, SpeciesLabel, parse_label

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
LABELS = ROOT / "models" / "inat_bird_labels.txt"
FIXTURES = Path(__file__).parent / "fixtures"


def rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def test_parse_label_latin_and_common():
    assert parse_label("Cyanistes caeruleus (Eurasian Blue Tit)") == SpeciesLabel("Cyanistes caeruleus", "Eurasian Blue Tit")


def test_parse_label_background():
    assert parse_label("background") == SpeciesLabel("background", "background")


def test_parse_label_nested_parentheses():
    assert parse_label("Circus cyaneus hudsonius (Northern Harrier (American))") == SpeciesLabel(
        "Circus cyaneus hudsonius", "Northern Harrier (American)"
    )


def test_parse_label_without_parentheses_and_whitespace():
    assert parse_label("  Passer domesticus \n") == SpeciesLabel("Passer domesticus", "Passer domesticus")


# Requires the downloaded model; only used by the skipif'd TestRealModel class below.
@pytest.fixture(scope="module")
def clf():
    return InatBirdClassifier(MODEL, LABELS)


@pytest.mark.skipif(not MODEL.exists(), reason="species model not downloaded")
class TestRealModel:
    def test_labels_match_output_size(self, clf):
        assert len(clf.labels) == 965
        assert any(l.latin == "background" for l in clf.labels)

    def test_blue_tit_is_blue_tit(self, clf):
        label, p = clf.top(rgb(FIXTURES / "blue_tit.jpg"), k=1)[0]
        assert label.common == "Eurasian Blue Tit"
        assert p >= 0.7

    def test_no_resoftmax(self, clf):
        # A re-softmax of an already-normalised vector would flatten everything to ~1/965.
        probs = [p for _, p in clf.predict(rgb(FIXTURES / "blue_tit.jpg"))]
        assert max(probs) > 0.5
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert 0.5 <= sum(probs) <= 1.01  # uint8 quantisation drops tiny probabilities, so < 1 is normal

    def test_cat_is_uncertain(self, clf):
        _, p = clf.top(rgb(FIXTURES / "cat.jpg"), k=1)[0]
        assert p < 0.5

    def test_top_skips_background_by_default(self, clf):
        top = clf.top(rgb(FIXTURES / "cat.jpg"), k=5)
        assert all(l.latin != "background" for l, _ in top)
        assert len(top) == 5
        full = clf.predict(rgb(FIXTURES / "cat.jpg"))
        assert len(full) == 965
