import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.species import (
    CLIP_MEAN,
    CLIP_STD,
    Bank,
    BioClipSpecies,
    InatBirdClassifier,
    SpeciesLabel,
    load_bank,
    model_id_of,
    parse_label,
)

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


BIOCLIP = ROOT / "models" / "bioclip2-vision-int8.onnx"
BANK = ROOT / "models" / "bioclip2-birds.npz"
D = 8  # embedding size of the stand-in


def write_standin_onnx(path: Path, d: int = D, seed: int = 0) -> np.ndarray:
    """pixels[1,3,224,224] -> per-channel mean (3) -> MatMul W[3,d] -> embedding[1,d]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w = rng.standard_normal((3, d)).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["pixels"], ["m"], axes=[2, 3], keepdims=0),  # [1,3]
        helper.make_node("MatMul", ["m", "W"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes, "standin",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, d])],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return w


def write_bank(path: Path, emb: np.ndarray, model_id: str) -> None:
    np.savez(
        path,
        ids=np.array([11, 22, 33], np.int32),
        latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Passer domesticus"]),
        common=np.array(["Black Redstart", "European Robin", "House Sparrow"]),
        emb=emb.astype(np.float16),
        logit_scale=np.float32(100.0),
        model=np.str_(model_id),
        prompt=np.str_("a photo of {latin} ({common})."),
    )


def emb_of(rgb, w):
    px = (np.array(rgb, np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    e = px @ w
    return e / np.linalg.norm(e)


@pytest.fixture
def standin(tmp_path):
    w = write_standin_onnx(tmp_path / "m.onnx")
    # bank rows = the embeddings of pure red / green / blue images, so a flat image of that
    # colour must win.
    bank = np.vstack([emb_of((255, 0, 0), w), emb_of((0, 255, 0), w), emb_of((0, 0, 255), w)])
    write_bank(tmp_path / "b.npz", bank, model_id_of(tmp_path / "m.onnx"))
    return tmp_path / "m.onnx", tmp_path / "b.npz"


def flat(rgb, h=100, w=120):
    return np.full((h, w, 3), rgb, np.uint8)


def test_model_id_is_sha_prefixed(tmp_path):
    write_standin_onnx(tmp_path / "m.onnx")
    mid = model_id_of(tmp_path / "m.onnx")
    assert mid == "bioclip-2/int8/" + hashlib.sha256((tmp_path / "m.onnx").read_bytes()).hexdigest()[:12]


def test_load_bank_shapes(standin):
    _, bank_path = standin
    b = load_bank(bank_path)
    assert isinstance(b, Bank) and b.ids.tolist() == [11, 22, 33] and b.emb.shape == (3, D) and b.emb.dtype == np.float32
    assert b.model.startswith("bioclip-2/int8/") and b.logit_scale == 100.0
    assert b.resolve("black redstart") == 11 and b.resolve("ERITHACUS RUBECULA") == 22 and b.resolve("Dodo") is None


def test_load_bank_rejects_bad_files(tmp_path):
    (tmp_path / "x.npz").write_bytes(b"nope")
    with pytest.raises(ValueError, match="bank"):
        load_bank(tmp_path / "x.npz")
    np.savez(tmp_path / "y.npz", ids=np.array([1]))
    with pytest.raises(ValueError, match="bank"):
        load_bank(tmp_path / "y.npz")


def test_top_over_whole_bank(standin):
    sp = BioClipSpecies(*standin)
    top = sp.top(flat((255, 0, 0)), k=3)
    assert len(top) == 3 and top[0][0] == SpeciesLabel("Phoenicurus ochruros", "Black Redstart")
    assert top[0][1] > 0.9 and abs(sum(p for _, p in top) - 1.0) < 1e-4
    assert sp.candidate_count == 3


def test_k1_returns_one_entry(standin):
    sp = BioClipSpecies(*standin)
    assert len(sp.top(flat((0, 0, 255)), k=1)) == 1


def test_tiny_crop_goes_through(standin):
    sp = BioClipSpecies(*standin)
    assert len(sp.top(flat((0, 255, 0), h=8, w=8))) == 3


def test_red_is_not_read_as_blue(standin):
    # Pins RGB order: a BGR mix-up would turn pure red into pure blue (the House Sparrow row).
    sp = BioClipSpecies(*standin)
    assert sp.top(flat((255, 0, 0)))[0][0].common != "House Sparrow"


def test_candidates_restrict_the_softmax(standin):
    sp = BioClipSpecies(*standin)
    assert sp.set_candidates([22, 33]) == 2 and sp.candidate_count == 2
    top = sp.top(flat((255, 0, 0)), k=5)
    assert len(top) == 2 and all(l.common != "Black Redstart" for l, _ in top)
    assert abs(sum(p for _, p in top) - 1.0) < 1e-4
    assert sp.set_candidates(None) == 3


def test_unknown_candidate_ids_are_ignored_and_empty_means_no_species(standin):
    sp = BioClipSpecies(*standin)
    assert sp.set_candidates([99, 11]) == 1
    assert sp.top(flat((0, 255, 0)))[0][0].common == "Black Redstart"
    assert sp.set_candidates([]) == 0
    assert sp.top(flat((0, 255, 0))) == []


def test_model_id_mismatch_raises(standin, tmp_path):
    model, _ = standin
    write_bank(tmp_path / "other.npz", np.eye(3, D, dtype=np.float32), model_id="bioclip-2/int8/deadbeef0000")
    with pytest.raises(ValueError, match="bank"):
        BioClipSpecies(model, tmp_path / "other.npz")
    BioClipSpecies(model, tmp_path / "other.npz", check_model=False)   # escape hatch


def test_common_name_falls_back_to_latin(standin, tmp_path):
    model, _ = standin
    w = write_standin_onnx(tmp_path / "m2.onnx")
    np.savez(tmp_path / "c.npz", ids=np.array([5], np.int32), latin=np.array(["Upupa epops"]), common=np.array([""]),
             emb=emb_of((255, 0, 0), w).astype(np.float16)[None], logit_scale=np.float32(100.0),
             model=np.str_(model_id_of(tmp_path / "m2.onnx")), prompt=np.str_("p"))
    sp = BioClipSpecies(tmp_path / "m2.onnx", tmp_path / "c.npz")
    assert sp.top(flat((255, 0, 0)))[0][0] == SpeciesLabel("Upupa epops", "Upupa epops")


def test_preprocess_shape_and_normalisation():
    from birdy.species import preprocess
    x = preprocess(flat((255, 255, 255), h=50, w=300))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.allclose(x[0, :, 0, 0], (1.0 - CLIP_MEAN) / CLIP_STD, atol=1e-5)


@pytest.mark.skipif(not (BIOCLIP.exists() and BANK.exists()), reason="BioCLIP model/bank not built")
class TestRealBioClip:
    def test_bird_fixture_gets_a_bird_name(self):
        sp = BioClipSpecies(BIOCLIP, BANK)
        label, p = sp.top(rgb(FIXTURES / "bird.jpg"), k=1)[0]
        assert p > 0.3 and label.latin and label.common
