# Species Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every capture is also run through the Coral iNat bird species model; confident species are stored and shown, gate misses can be rescued by a confident species, a `/species` summary page and species filter answer "what visited?", and an upload box classifies a phone photo on demand.

**Architecture:** A `TwoStageClassifier` implements the existing `Classifier` protocol: it decodes/crops once, runs the ImageNet gate (`predict`) and the new `InatBirdClassifier` (`top`) on the same pixels, and combines them with two live-tunable thresholds. The pipeline is unchanged for live captures; its sweep additionally drains a `species_checked = 0` backlog. The store gains a migration, a species filter and a summary query; the web gains a dropdown, `/species` and `POST /classify`.

**Tech Stack:** existing stack; new model `models/mobilenet_v2_1.0_224_inat_bird_quant.tflite` + `models/inat_bird_labels.txt` (Coral test_data, Apache-2.0, no login).

**Spec:** `docs/superpowers/specs/2026-10-03-species-recognition-design.md`
**Branch:** `species` (created, spec committed).
**Model dispatch hint:** `[opus-ok]` = mechanical.

---

## File structure

| File | Responsibility |
|---|---|
| `birdy/tflite.py` (new) | `load_interpreter()` shared by both models |
| `birdy/species.py` (new) | `SpeciesLabel`, `parse_label`, `InatBirdClassifier` |
| `birdy/classifier.py` (modify) | `CropLoader`, `ImageNetBirdClassifier.predict`, `TwoStageClassifier` |
| `birdy/config.py` (modify) | `SpeciesConfig` + `[species]` parsing/validation |
| `birdy/settings.py` (modify) | two new patch fields |
| `birdy/store.py` (modify) | migration, `species_checked`, backlog, summary, species filter, `last_species` |
| `birdy/pipeline.py` (modify) | backlog drain in `sweep()` |
| `birdy/web/app.py` (modify) | dropdown, `/species`, `POST /classify`, `last_species` |
| `birdy/web/templates/species.html` (new), `index.html`, `settings.html` (modify) | |
| `birdy/cli.py` (modify) | two-stage wiring; `classify` prints species |
| `models/download.sh`, `models/README.md`, `models/LICENSE` (modify); model + labels (new) | |
| `tests/fixtures/blue_tit.jpg` (new) | phone-screen blue tit crop from the Pi (scores 0.85) |
| `tests/test_species.py`, `tests/test_two_stage.py` (new); `test_classifier.py`, `test_config.py`, `test_settings.py`, `test_store.py`, `test_pipeline.py`, `test_web.py`, `test_cli.py`, `conftest.py` (modify) | |

---

### Task 1: Model assets, fixture, `tflite.py`, `species.py` `[opus-ok]`

**Files:**
- Create: `birdy/tflite.py`, `birdy/species.py`, `tests/test_species.py`, `tests/fixtures/blue_tit.jpg`, `models/mobilenet_v2_1.0_224_inat_bird_quant.tflite`, `models/inat_bird_labels.txt`
- Modify: `models/download.sh`, `models/README.md`, `models/LICENSE`

- [ ] **Step 1: Assets**

Append to `models/download.sh` (before the final `echo`):
```bash
coral=https://raw.githubusercontent.com/google-coral/test_data/master
curl -sSL -o mobilenet_v2_1.0_224_inat_bird_quant.tflite "$coral/mobilenet_v2_1.0_224_inat_bird_quant.tflite"
curl -sSL -o inat_bird_labels.txt "$coral/inat_bird_labels.txt"
echo "species model: $(du -h mobilenet_v2_1.0_224_inat_bird_quant.tflite | cut -f1), labels: $(wc -l < inat_bird_labels.txt) lines"
```
Run `models/download.sh` → the last line reads `species model: 3.5M, labels: 965 lines` (first label `background`? No — `background` is the **last** line; check: `tail -1 models/inat_bird_labels.txt` → `background`. If it is elsewhere, don't worry: the code finds it by name).

`models/README.md`: add a paragraph: "`mobilenet_v2_1.0_224_inat_bird_quant.tflite` + `inat_bird_labels.txt` are the Coral iNaturalist bird classifier (965 classes incl. `background`, MobileNetV2 uint8, output already softmax-scaled 1/256) from github.com/google-coral/test_data, Apache-2.0. Used for species recognition." `models/LICENSE`: add a line naming those two files under Apache-2.0 with the repo URL.

Fixture: `cp /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/blue_tit.jpg tests/fixtures/blue_tit.jpg` (640×437, ~41 KB; a crop of the Pi's capture of a phone showing a blue tit). Add a line to `tests/fixtures/download.sh`'s header comment: "`blue_tit.jpg` is a crop of a real Pi capture (committed, no download)."

- [ ] **Step 2: Write the failing tests**

`tests/test_species.py`:
```python
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


def test_parse_label_without_parentheses_and_whitespace():
    assert parse_label("  Passer domesticus \n") == SpeciesLabel("Passer domesticus", "Passer domesticus")


@pytest.mark.skipif(not MODEL.exists(), reason="species model not downloaded")
class TestRealModel:
    @pytest.fixture(scope="class")
    def clf(self):
        return InatBirdClassifier(MODEL, LABELS)

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
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_species.py -q` → FAIL, `ModuleNotFoundError: No module named 'birdy.species'`.

- [ ] **Step 4: Write `birdy/tflite.py`**

```python
"""Shared TFLite interpreter loading (ai-edge-litert, with the old tflite-runtime as fallback)."""

from __future__ import annotations

from pathlib import Path


def load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older Pi installs
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=str(model_path))
```

- [ ] **Step 5: Write `birdy/species.py`**

```python
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
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_species.py -q` → `8 passed`.

- [ ] **Step 7: Commit**

```bash
git add birdy/tflite.py birdy/species.py tests/test_species.py tests/fixtures/blue_tit.jpg tests/fixtures/download.sh models/
git commit -m "Add iNat bird species model and InatBirdClassifier"
```

---

### Task 2: `[species]` config + settings fields `[opus-ok]`

**Files:**
- Modify: `birdy/config.py`, `birdy.toml.example`, `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` (settings_context only), `tests/test_config.py`, `tests/test_settings.py`, `tests/test_web.py` (form helper only)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:
```python
def test_species_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.species.model == tmp_path / "models" / "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
    assert cfg.species.labels == tmp_path / "models" / "inat_bird_labels.txt"
    assert cfg.species.threshold == 0.5 and cfg.species.rescue_threshold == 0.7


def test_species_rescue_must_not_be_below_threshold(tmp_path):
    with pytest.raises(ConfigError, match="species.rescue_threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\nthreshold = 0.8\nrescue_threshold = 0.6\n"))


@pytest.mark.parametrize("key", ["threshold", "rescue_threshold"])
def test_species_thresholds_range(tmp_path, key):
    with pytest.raises(ConfigError, match=f"species.{key}"):
        load_config(write(tmp_path, MINIMAL + f"\n[species]\n{key} = 1.5\n"))
```

In `tests/test_settings.py`: `default_patch` gains `species_threshold=config.species.threshold, species_rescue_threshold=config.species.rescue_threshold`; `FORM` gains `"species_threshold": "0.5", "species_rescue_threshold": "0.7"`; `FULL_TOML` gains this section (non-default values so the round-trip test can't pass trivially):
```toml
[species]
model = "models/other_species.tflite"
labels = "models/other_species_labels.txt"
threshold = 0.4
rescue_threshold = 0.9
```
Add:
```python
def test_species_thresholds_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, species_threshold=0.3, species_rescue_threshold=0.8))
    assert new.species.threshold == 0.3 and new.species.rescue_threshold == 0.8


def test_species_rescue_below_threshold_rejected(config_path, config):
    with pytest.raises(ConfigError, match="species.rescue_threshold"):
        save_settings(config_path, config, default_patch(config, species_threshold=0.9, species_rescue_threshold=0.5))


def test_parse_form_species_fields():
    p = parse_settings_form({**FORM, "species_threshold": "0.35", "species_rescue_threshold": "0.75"})
    assert p.species_threshold == 0.35 and p.species_rescue_threshold == 0.75
```
In `tests/test_web.py`: `post_settings()`'s form dict gains the same two keys; add:
```python
def test_settings_page_has_species_rows(client):
    r = client.get("/settings").text
    assert 'name="species_threshold"' in r and 'name="species_rescue_threshold"' in r
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_web.py -q` → failures: `AttributeError: 'Config' object has no attribute 'species'`, `TypeError: SettingsPatch.__init__() got an unexpected keyword argument`, `test_config_to_dict_covers_every_field` (once `species` exists), web 422s.

- [ ] **Step 3: `birdy/config.py`**

Add after `ClassifierConfig`:
```python
@dataclass(frozen=True)
class SpeciesConfig:
    model: Path
    labels: Path
    threshold: float          # show a species at or above this
    rescue_threshold: float   # promote a gate-rejected frame to "bird" at or above this
```
Add `species: SpeciesConfig` to `Config` right after `classifier`. In `load_config`, after the classifier block:
```python
    sp = raw.get("species", {})
    species_threshold = _num(sp, "species", "threshold", 0.5, float)
    _check(0.0 <= species_threshold <= 1.0, "species.threshold must be between 0 and 1")
    rescue_threshold = _num(sp, "species", "rescue_threshold", 0.7, float)
    _check(0.0 <= rescue_threshold <= 1.0, "species.rescue_threshold must be between 0 and 1")
    _check(rescue_threshold >= species_threshold, "species.rescue_threshold must be >= species.threshold")
```
and in the `Config(...)` construction, after `classifier=...`:
```python
        species=SpeciesConfig(
            model=_path(base, sp, "species", "model", "models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"),
            labels=_path(base, sp, "species", "labels", "models/inat_bird_labels.txt"),
            threshold=species_threshold,
            rescue_threshold=rescue_threshold,
        ),
```
`birdy.toml.example`, after `[classifier]`:
```toml
[species]
model = "models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"
labels = "models/inat_bird_labels.txt"
threshold = 0.5            # show a species name at or above this confidence
rescue_threshold = 0.7     # a frame the gate rejected becomes a bird if the species model is this sure
```

- [ ] **Step 4: `birdy/settings.py`**

`FIELD_KEYS`: add `"species_threshold": "species.threshold", "species_rescue_threshold": "species.rescue_threshold"`. `SettingsPatch`: add `species_threshold: float` and `species_rescue_threshold: float` after `classifier_threshold`. `config_to_dict`: add
```python
        "species": {
            "model": str(config.species.model),
            "labels": str(config.species.labels),
            "threshold": config.species.threshold,
            "rescue_threshold": config.species.rescue_threshold,
        },
```
`apply_patch`: `data["species"]["threshold"] = patch.species_threshold; data["species"]["rescue_threshold"] = patch.species_rescue_threshold`. `parse_settings_form`: `species_threshold=_num(form, "species_threshold", float), species_rescue_threshold=_num(form, "species_rescue_threshold", float)`.

- [ ] **Step 5: Template + context**

`settings.html`, after the classifier row:
```html
{{ num("species_threshold", "species threshold", 0, 1, 0.01) }}
{{ num("species_rescue_threshold", "species rescue threshold", 0, 1, 0.01) }}
```
`app.py` `settings_context` defaults dict: add `"species_threshold": cfg.species.threshold, "species_rescue_threshold": cfg.species.rescue_threshold,`; `readonly` gains `"species.model": str(cfg.species.model)`.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_web.py -q` → all pass. Full suite: `.venv/bin/pytest -q` green.

- [ ] **Step 7: Commit**

```bash
git add birdy/config.py birdy.toml.example birdy/settings.py birdy/web/templates/settings.html birdy/web/app.py tests/test_config.py tests/test_settings.py tests/test_web.py
git commit -m "Add [species] config and settings-page thresholds"
```

---

### Task 3: Classifier refactor + `TwoStageClassifier` + CLI

**Files:**
- Modify: `birdy/classifier.py`, `birdy/cli.py`, `tests/test_classifier.py`, `tests/test_cli.py`
- Create: `tests/test_two_stage.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_two_stage.py`:
```python
import dataclasses
import logging
from pathlib import Path

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

    p = tmp_path / "x.jpg"
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
```

Append to `tests/test_classifier.py` inside `TestRealModel`:
```python
    def test_predict_on_decoded_image(self, clf):
        import cv2

        img = cv2.cvtColor(cv2.imread(str(FIXTURES / "bird.jpg")), cv2.COLOR_BGR2RGB)
        assert clf.predict(img) > 0.5
```

`tests/test_cli.py`: `test_classify_command_prints_prediction` additionally asserts `"species (iNat):" in out` and `"gate (ImageNet):" in out`. In `test_cmd_run_passes_live_everywhere`, the `_classifier` stub must now accept `(config, live)`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_two_stage.py tests/test_classifier.py tests/test_cli.py -q` → `ImportError: cannot import name 'CropLoader'` etc.

- [ ] **Step 3: Rewrite `birdy/classifier.py`** (keep `Prediction`, `Classifier`, `load_bird_indices`, `bird_probability`, `_softmax`, `crop_to_bbox` as they are; replace `_load_interpreter` with the shared one; add `CropLoader`; restructure `ImageNetBirdClassifier`; add `TwoStageClassifier`)

```python
"""Bird / not-bird gate (ImageNet MobileNetV2) and the two-stage classifier that adds species."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from birdy.live import LiveConfig
from birdy.motion import BBox
from birdy.species import InatBirdClassifier, SpeciesLabel
from birdy.tflite import load_interpreter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    confidence: float  # P(bird), 0..1
    species: str | None = None
    species_confidence: float | None = None


class Classifier(Protocol):
    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction: ...


# load_bird_indices, bird_probability, _softmax, crop_to_bbox: unchanged from the current file.


class CropLoader:
    """Decode a still and crop it to the ROI bbox (lores coords -> main coords), as RGB."""

    def __init__(self, lores_size: tuple[int, int], pad: float = 0.1) -> None:
        self._lores_size = lores_size
        self._pad = pad
        self._warned_bbox: BBox | None = None

    def load(self, image_path: Path, roi_bbox: BBox | None) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        if roi_bbox is not None:
            crop = crop_to_bbox(img, roi_bbox, self._lores_size, self._pad)
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
        out = (out.astype(np.float32) - zero_point) * scale if scale else out.astype(np.float32)
        # The hosted mobilenet_v2_1.0_224_quant model has no softmax layer: its
        # output dequantizes to logits (scale ~0.099, zero point 58).
        return _softmax(out)

    def predict(self, img_rgb: np.ndarray) -> float:
        """P(bird) for an already-decoded RGB image."""
        return bird_probability(self._probs(img_rgb), self._bird_indices)

    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction:
        """`threshold` overrides the constructor default so it can be tuned live."""
        t = self._threshold if threshold is None else threshold
        conf = self.predict(self._crops.load(image_path, roi_bbox))
        return Prediction(is_bird=conf >= t, confidence=conf)

    def top_k(self, image_path: Path, roi_bbox: BBox | None, k: int = 5) -> list[tuple[str, float]]:
        probs = self._probs(self._crops.load(image_path, roi_bbox))
        order = np.argsort(probs)[::-1][:k]
        return [(self._labels[i] if i < len(self._labels) else str(i), float(probs[i])) for i in order]


class TwoStageClassifier:
    """Gate (bird / not-bird) + species on the same decoded crop. Implements `Classifier`.

    Rules (thresholds from `live`): a species is reported when its probability meets
    `species.threshold`; a gate-rejected frame is *rescued* (is_bird=True) when the
    species probability meets the higher `species.rescue_threshold`.
    """

    def __init__(
        self,
        gate: ImageNetBirdClassifier,
        species: InatBirdClassifier,
        live: LiveConfig,
        lores_size: tuple[int, int],
        pad: float = 0.1,
    ) -> None:
        self.gate = gate
        self.species = species
        self._live = live
        self._crops = CropLoader(lores_size, pad)
        self._species_errors: set[str] = set()

    def classify_image(
        self, img_rgb: np.ndarray, threshold: float | None = None
    ) -> tuple[Prediction, list[tuple[SpeciesLabel, float]]]:
        cfg = self._live.get()
        t_gate = cfg.classifier.threshold if threshold is None else threshold
        t_sp, t_rescue = cfg.species.threshold, cfg.species.rescue_threshold

        p_bird = self.gate.predict(img_rgb)
        try:
            top = self.species.top(img_rgb, k=5)
        except Exception as e:
            key = f"{type(e).__name__}: {e}"
            if key not in self._species_errors:  # log each distinct failure once, not per capture
                log.exception("species model failed (%s); continuing with the gate only", key)
                self._species_errors.add(key)
            top = []
        label, p_sp = top[0] if top else (None, 0.0)

        species = label.common if label is not None and p_sp >= t_sp else None
        rescued = p_bird < t_gate and p_sp >= t_rescue
        is_bird = p_bird >= t_gate or rescued
        confidence = max(p_bird, p_sp) if rescued else p_bird
        pred = Prediction(
            is_bird=is_bird,
            confidence=confidence,
            species=species,
            species_confidence=p_sp if species is not None else None,
        )
        return pred, top

    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction:
        return self.classify_image(self._crops.load(image_path, roi_bbox), threshold)[0]
```

- [ ] **Step 4: `birdy/cli.py`**

```python
from birdy.classifier import CropLoader, ImageNetBirdClassifier, TwoStageClassifier
from birdy.live import LiveConfig
from birdy.species import InatBirdClassifier
...
def _classifier(config: Config, live: LiveConfig) -> TwoStageClassifier:
    gate = ImageNetBirdClassifier(
        config.classifier.model,
        config.classifier.labels,
        config.classifier.bird_classes,
        threshold=config.classifier.threshold,
        lores_size=config.camera.lores_size,
    )
    species = InatBirdClassifier(config.species.model, config.species.labels)
    return TwoStageClassifier(gate, species, live, lores_size=config.camera.lores_size)
```
In `cmd_run`: create `live = LiveConfig(config)` first (it already does), then `pipeline = Pipeline(store, _classifier(config, live), live)`. Remove the now-duplicate `from birdy.live import LiveConfig` inside `cmd_run` if it's imported at module level.

```python
def cmd_classify(config: Config, image: str) -> int:
    live = LiveConfig(config)
    clf = _classifier(config, live)
    img = CropLoader(config.camera.lores_size).load(Path(image), None)
    pred, top = clf.classify_image(img)
    print(f"bird={pred.is_bird} p_bird={pred.confidence:.3f} species={pred.species or '-'}")
    print("gate (ImageNet):")
    for label, prob in clf.gate.top_k(Path(image), None, k=5):
        print(f"  {prob:.3f}  {label}")
    print("species (iNat):")
    for label, prob in top:
        print(f"  {prob:.3f}  {label.common} ({label.latin})")
    return 0
```
A missing species model file raises `ValueError`/`OSError` from the interpreter inside `_classifier`; in `main()`, wrap the `run` and `classify` branches so `(OSError, ValueError)` from classifier construction print `model error: …` to stderr and return 2 (exit 2 = config-class error, no restart loop).

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_two_stage.py tests/test_classifier.py tests/test_cli.py -q` → all pass (classifier 16, two-stage 9, cli 8). Full suite green; pyflakes silent.

- [ ] **Step 6: Commit**

```bash
git add birdy/classifier.py birdy/cli.py tests/test_two_stage.py tests/test_classifier.py tests/test_cli.py
git commit -m "Add TwoStageClassifier (gate + species) and wire it into the CLI"
```

---

### Task 4: Store — migration, backlog, summary, species filter `[opus-ok]`

**Files:**
- Modify: `birdy/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_store.py`)

```python
import sqlite3

from birdy.store import SpeciesRow


def add_species(store, when, species, conf, is_bird=True):
    cid = add(store, when)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return cid


def test_migration_adds_species_checked_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending');"
        "INSERT INTO captures (captured_at, path, motion_score, is_bird, status) VALUES ('2026-10-03T09:00:00.000+00:00', 'a.jpg', 1, 1, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    assert [c.id for c in s.species_backlog()] == [1]
    assert s.get(1).species_checked is False
    s.close()
    Store(data).close()  # second open: migration is a no-op


def test_update_classification_marks_species_checked(store):
    cid = add(store, T0)
    assert store.get(cid).species_checked is False
    store.update_classification(cid, True, 0.9, None, None)
    assert store.get(cid).species_checked is True
    assert store.species_backlog() == []


def test_species_backlog_newest_first_and_limit(store):
    ids = [add_species(store, T0 + timedelta(minutes=i), None, None) for i in range(3)]
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    assert [c.id for c in store.species_backlog()] == ids[::-1]
    assert [c.id for c in store.species_backlog(limit=1)] == [ids[2]]


def test_mark_species_checked(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.9, None, None)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    store.mark_species_checked(cid)
    assert store.species_backlog() == []
    assert store.get(cid).status == "done"


def test_species_summary(store):
    a = add_species(store, T0, "Robin", 0.6)
    b = add_species(store, T0 + timedelta(minutes=1), "Robin", 0.9)
    c = add_species(store, T0 + timedelta(minutes=2), "Blue Tit", 0.7)
    add_species(store, T0 + timedelta(minutes=3), None, None)
    rows = store.species_summary()
    assert [r.species for r in rows] == ["Blue Tit", "Robin"]  # last seen desc
    robin = rows[1]
    assert isinstance(robin, SpeciesRow)
    assert robin.count == 2 and robin.first_seen == T0 and robin.last_seen == T0 + timedelta(minutes=1)
    assert robin.best_capture_id == b and robin.best_thumb_path == store.get(b).thumb_path
    assert rows[0].best_capture_id == c


def test_species_summary_empty(store):
    assert store.species_summary() == []


def test_list_and_count_by_species(store):
    r1 = add_species(store, T0, "Robin", 0.9)
    add_species(store, T0 + timedelta(minutes=1), "Blue Tit", 0.9)
    assert [x.id for x in store.list_captures("all", species="Robin")] == [r1]
    assert store.count("all", species="Robin") == 1
    assert store.count("birds", species="Blue Tit") == 1
    assert store.count("non-birds", species="Blue Tit") == 0


def test_last_species(store):
    assert store.last_species() is None
    add_species(store, T0, "Robin", 0.9)
    add_species(store, T0 + timedelta(minutes=5), None, None)
    name, when = store.last_species()
    assert name == "Robin" and when == T0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_store.py -q` → `ImportError: cannot import name 'SpeciesRow'`.

- [ ] **Step 3: Modify `birdy/store.py`**

Add index to `SCHEMA` (after the status index): `CREATE INDEX IF NOT EXISTS captures_species ON captures(species, captured_at DESC);` — but the column `species_checked` must NOT be in the `CREATE TABLE` (old DBs lack it; the migration adds it uniformly). In `__init__` after `executescript(SCHEMA)`, inside the same lock/transaction block:
```python
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(captures)")}
            if "species_checked" not in cols:
                self._conn.execute("ALTER TABLE captures ADD COLUMN species_checked INTEGER NOT NULL DEFAULT 0")
```
`Capture` gains `species_checked: bool` (last field); `_row_to_capture` adds `species_checked=bool(row["species_checked"])`.

Add after `Capture`:
```python
@dataclass(frozen=True)
class SpeciesRow:
    species: str
    count: int
    first_seen: datetime
    last_seen: datetime
    best_capture_id: int
    best_path: str
    best_thumb_path: str | None
```
`update_classification`: SQL becomes `UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ?, species_checked = 1, status = 'done' WHERE id = ?`.

New methods (in the captures section):
```python
    def species_backlog(self, limit: int = 50) -> list[Capture]:
        """Classified rows that predate species recognition, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM captures WHERE status = 'done' AND species_checked = 0"
                " ORDER BY captured_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def mark_species_checked(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET species_checked = 1 WHERE id = ?", (capture_id,))

    def species_summary(self) -> list[SpeciesRow]:
        with self._lock:
            groups = self._conn.execute(
                "SELECT species, COUNT(*) AS n, MIN(captured_at) AS first_seen, MAX(captured_at) AS last_seen"
                " FROM captures WHERE species IS NOT NULL GROUP BY species ORDER BY last_seen DESC"
            ).fetchall()
            out = []
            for g in groups:
                best = self._conn.execute(
                    "SELECT id, path, thumb_path FROM captures WHERE species = ?"
                    " ORDER BY species_confidence DESC, captured_at DESC LIMIT 1",
                    (g["species"],),
                ).fetchone()
                out.append(
                    SpeciesRow(
                        species=g["species"],
                        count=int(g["n"]),
                        first_seen=_from_db_time(g["first_seen"]),
                        last_seen=_from_db_time(g["last_seen"]),
                        best_capture_id=int(best["id"]),
                        best_path=best["path"],
                        best_thumb_path=best["thumb_path"],
                    )
                )
        return out

    def last_species(self) -> tuple[str, datetime] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT species, captured_at FROM captures WHERE species IS NOT NULL"
                " ORDER BY captured_at DESC LIMIT 1"
            ).fetchone()
        return None if row is None else (row["species"], _from_db_time(row["captured_at"]))
```
`list_captures` and `count` gain `species: str | None = None`:
```python
    def _where(self, filter: str, species: str | None) -> tuple[str, list]:
        where, params = FILTERS[filter], []
        if species is not None:
            where += " AND species = ?"
            params.append(species)
        return where, params

    def list_captures(self, filter: str = "all", limit: int = 50, offset: int = 0, species: str | None = None) -> list[Capture]:
        where, params = self._where(filter, species)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM captures WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def count(self, filter: str = "all", species: str | None = None) -> int:
        where, params = self._where(filter, species)
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM captures WHERE {where}", params).fetchone()[0])
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_store.py -q` → all pass (16 existing + 8). Full suite green (other tests use `Capture` via the store, so the new field is harmless).

- [ ] **Step 5: Commit**

```bash
git add birdy/store.py tests/test_store.py
git commit -m "Store: species_checked migration, backlog, summary, species filter"
```

---

### Task 5: Pipeline backlog drain `[opus-ok]`

**Files:**
- Modify: `birdy/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`)

```python
def unchecked(store, ids):
    with store._lock, store._conn:
        store._conn.execute(f"UPDATE captures SET species_checked = 0 WHERE id IN ({','.join('?' * len(ids))})", ids)


class SpeciesFake(FakeClassifier):
    def classify(self, image_path, roi_bbox, threshold=None):
        pred = super().classify(image_path, roi_bbox, threshold)
        return Prediction(is_bird=True, confidence=0.9, species="Robin", species_confidence=0.8)


def test_sweep_drains_species_backlog(store, live):
    evs = [add(store, i) for i in range(3)]
    for ev in evs:
        store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id for ev in evs])
    p = Pipeline(store, SpeciesFake(), live)
    p.sweep()
    assert store.species_backlog() == []
    assert all(store.get(ev.capture_id).species == "Robin" for ev in evs)
    assert all(store.get(ev.capture_id).status == "done" for ev in evs)


def test_backlog_failure_is_marked_checked_not_error(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p = Pipeline(store, FakeClassifier(fail_on={ev.path.name}), live)
    p.sweep()
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.species_checked is True and c.species is None


def test_backlog_respects_stop(store, live):
    evs = [add(store, i) for i in range(2)]
    for ev in evs:
        store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id for ev in evs])
    p = Pipeline(store, SpeciesFake(), live)
    p._stop.set()
    p.sweep()
    assert len(store.species_backlog()) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pipeline.py -q` → the three new tests fail (`species_backlog` still has rows / species None).

- [ ] **Step 3: Modify `birdy/pipeline.py`**

Add a constant `SPECIES_BACKLOG_PER_SWEEP = 50` and extend `sweep()`:
```python
    def sweep(self) -> None:
        for c in self._store.pending():
            if self._stop.is_set():
                break
            current = self._store.get(c.id)
            if current is None or current.status != "pending":
                continue
            self.process(c.id, self._store.photos_dir / c.path)
        # Rows classified before species recognition existed: re-run the (two-stage)
        # classifier so they get a species. Bounded per sweep to keep the Pi responsive.
        for c in self._store.species_backlog(limit=SPECIES_BACKLOG_PER_SWEEP):
            if self._stop.is_set():
                break
            self._backfill_species(c.id, self._store.photos_dir / c.path)

    def _backfill_species(self, capture_id: int, path: Path) -> None:
        cfg = self._live.get()
        try:
            pred = self._classifier.classify(path, roi_bbox(cfg.motion.roi), threshold=cfg.classifier.threshold)
            self._store.update_classification(capture_id, pred.is_bird, pred.confidence, pred.species, pred.species_confidence)
            if pred.species:
                log.info("backfill capture %d: %s %.2f", capture_id, pred.species, pred.species_confidence)
        except Exception:
            log.exception("species backfill failed for capture %d; marking checked", capture_id)
            try:
                self._store.mark_species_checked(capture_id)
            except Exception:
                log.exception("mark_species_checked failed for capture %d", capture_id)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -q` → `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/pipeline.py tests/test_pipeline.py
git commit -m "Pipeline: backfill species for pre-existing captures during sweeps"
```

---

### Task 6: Web — species dropdown, `/species` page, `POST /classify`, `last_species`

**Files:**
- Modify: `birdy/web/app.py`, `birdy/web/templates/index.html`, `tests/test_web.py`
- Create: `birdy/web/templates/species.html`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web.py`; also change the `client` fixture)

```python
from birdy.classifier import Prediction
from birdy.species import SpeciesLabel

BLUE_TIT = SpeciesLabel("Cyanistes caeruleus", "Eurasian Blue Tit")


class FakeTwoStage:
    def __init__(self):
        self.images = []

    def classify_image(self, img, threshold=None):
        self.images.append(img.shape)
        pred = Prediction(is_bird=True, confidence=0.93, species="Eurasian Blue Tit", species_confidence=0.81)
        return pred, [(BLUE_TIT, 0.81), (SpeciesLabel("Parus major", "Great Tit"), 0.05)]


@pytest.fixture
def classifier():
    return FakeTwoStage()


@pytest.fixture
def client(store, state, live, config_path, classifier):
    return TestClient(create_app(store, state, live, config_path, classifier=classifier))


def test_index_species_dropdown_and_filter(client, store):
    robin = add(store, 0, True)       # add() labels birds "Robin"
    add(store, 1, False)
    r = client.get("/")
    assert '<select name="species"' in r.text and "Robin (1)" in r.text
    r2 = client.get("/?species=Robin")
    assert robin.thumb_path in r2.text and store.get(robin.id + 1).thumb_path not in r2.text
    assert client.get("/?species=Dodo").status_code == 400


def test_index_links_to_species_page(client):
    assert 'href="/species"' in client.get("/").text


def test_species_page_empty(client):
    r = client.get("/species")
    assert r.status_code == 200 and "No species yet" in r.text


def test_species_page_table_and_best_photo(client, store):
    a = add(store, 0, True)
    b = add(store, 1, True)
    r = client.get("/species").text
    assert "Robin" in r and ">2<" in r
    assert f"/photos/{store.get(b.id).path}" in r or f"/photos/{a.path}" in r  # best photo link present


def test_classify_upload_renders_result(client, classifier, store):
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    before = store.count("all")
    r = client.post("/classify", files={"image": ("phone.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    assert "Eurasian Blue Tit" in r.text and "81%" in r.text and "bird" in r.text
    assert classifier.images and classifier.images[0][2] == 3
    assert store.count("all") == before
    assert not any(store.photos_dir.rglob("*.jpg"))


def test_classify_upload_garbage_422(client):
    r = client.post("/classify", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422 and "could not decode" in r.text


def test_classify_upload_missing_422(client):
    assert client.post("/classify", data={}).status_code == 422


def test_classify_upload_too_large_413(client):
    big = b"\xff\xd8" + b"\0" * (10 * 1024 * 1024 + 1)
    assert client.post("/classify", files={"image": ("big.jpg", big, "image/jpeg")}).status_code == 413


def test_classify_without_classifier_503(store, state, live, config_path):
    c = TestClient(create_app(store, state, live, config_path))
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    assert c.post("/classify", files={"image": ("p.jpg", data, "image/jpeg")}).status_code == 503


def test_status_last_species(client, store):
    assert client.get("/status").json()["last_species"] is None
    add(store, 0, True)
    assert client.get("/status").json()["last_species"] == "Robin"
```
(`from pathlib import Path` at the top of the file if not already there.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_web.py -q` → `TypeError: create_app() got an unexpected keyword argument 'classifier'`.

- [ ] **Step 3: Modify `birdy/web/app.py`**

Signature: add `classifier: "TwoStageClassifier | None" = None` after `queue_depth` (import `from birdy.classifier import TwoStageClassifier` under `TYPE_CHECKING`, or just type as `object | None` with a docstring — keep the import light: use `from typing import TYPE_CHECKING`). Add `MAX_UPLOAD_BYTES = 10 * 1024 * 1024`.

`status_payload`: after `last = ...`:
```python
        ls = store.last_species()
        snap.update(..., last_species=None if ls is None else ls[0])
```
`index`:
```python
    def index(request: Request, filter: str = "all", page: int = 1, species: str | None = None):
        if filter not in FILTERS:
            raise HTTPException(status_code=400, detail="filter must be one of all, birds, non-birds")
        summary = store.species_summary()
        if species == "":
            species = None
        if species is not None and species not in {r.species for r in summary}:
            raise HTTPException(status_code=400, detail="unknown species")
        total = store.count(filter, species=species)
        pages = max(1, -(-total // PAGE_SIZE))
        page = min(max(1, page), pages)
        captures = store.list_captures(filter, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE, species=species)
        return templates.TemplateResponse(request, "index.html", {
            "captures": captures, "filter": filter, "page": page, "pages": pages, "total": total,
            "species": species, "species_rows": summary, "status": status_payload(),
        })
```
New routes:
```python
    def species_context(result: dict | None = None, error: str | None = None) -> dict:
        return {"rows": store.species_summary(), "result": result, "error": error, "status": status_payload()}

    @app.get("/species", response_class=HTMLResponse)
    def species_page(request: Request):
        return templates.TemplateResponse(request, "species.html", species_context())

    @app.post("/classify", response_class=HTMLResponse)
    async def classify_upload(request: Request):
        if classifier is None:
            raise HTTPException(status_code=503, detail="classifier not available")
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            return templates.TemplateResponse(request, "species.html", species_context(error="choose a photo first"), status_code=422)
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return templates.TemplateResponse(request, "species.html", species_context(error="photo larger than 10 MB"), status_code=413)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return templates.TemplateResponse(request, "species.html", species_context(error="could not decode that file as an image"), status_code=422)
        pred, top = classifier.classify_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = {
            "is_bird": pred.is_bird,
            "confidence": pred.confidence,
            "species": pred.species,
            "top": [(label.common, label.latin, p) for label, p in top[:3]],
            "filename": getattr(upload, "filename", "photo"),
        }
        return templates.TemplateResponse(request, "species.html", species_context(result=result))
```

- [ ] **Step 4: Templates**

`index.html`: in the header `.status` div add `<span><a href="/species">species</a></span>` before the settings link. Replace the `<nav>` block with:
```html
<nav>
  {% for f in ["all", "birds", "non-birds"] %}
    <a href="/?filter={{ f }}{% if species %}&species={{ species | urlencode }}{% endif %}" class="{{ 'active' if f == filter else '' }}">{{ f }}</a>
  {% endfor %}
  <select name="species" onchange="location.href='/?filter={{ filter }}&species=' + encodeURIComponent(this.value)">
    <option value="" {{ 'selected' if not species else '' }}>All species</option>
    {% for r in species_rows %}
    <option value="{{ r.species }}" {{ 'selected' if r.species == species else '' }}>{{ r.species }} ({{ r.count }})</option>
    {% endfor %}
  </select>
  <span>({{ total }})</span>
</nav>
```
Pager links: append `{% if species %}&species={{ species | urlencode }}{% endif %}` to both hrefs. Add `select { padding: 4px 6px; background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; }` to the CSS.

`species.html` (new):
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy species</title>
<style>
  :root { color-scheme: light dark; --bg: #fafaf7; --fg: #222; --muted: #777; --card: #fff; --line: #ddd; --ok: #2a7; --bad: #c66; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #999; --card: #222; --line: #333; } }
  body { margin: 0; padding: 16px; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); max-width: 820px; }
  a { color: var(--fg); }
  h1 { font-size: 20px; margin: 0 0 12px; } h2 { font-size: 15px; margin: 20px 0 8px; }
  table { border-collapse: collapse; width: 100%; } th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }
  td img { width: 96px; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 4px; display: block; }
  .hint { color: var(--muted); font-size: 12px; }
  .banner { padding: 8px 12px; border-radius: 6px; margin: 12px 0; }
  .banner.ok { background: #2a72; border: 1px solid var(--ok); } .banner.bad { background: #c662; border: 1px solid var(--bad); }
  button { padding: 8px 16px; border-radius: 6px; border: 1px solid var(--ok); background: var(--ok); color: #fff; font-weight: bold; cursor: pointer; }
</style>
</head>
<body>
<h1><a href="/">birdy</a> › species</h1>

<h2>Try a photo</h2>
<form method="post" action="/classify" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" capture="environment" required>
  <button type="submit">Classify</button>
  <div class="hint">Runs both models on the photo. Nothing is stored.</div>
</form>
{% if error %}<div class="banner bad">{{ error }}</div>{% endif %}
{% if result %}
<div class="banner ok">
  <b>{{ result.filename }}</b>:
  {% if result.is_bird %}bird ({{ (result.confidence * 100) | round(0) | int }}%){% else %}not a bird ({{ (result.confidence * 100) | round(0) | int }}% bird){% endif %}
  {% if result.species %} — <b>{{ result.species }}</b>{% endif %}
  <div class="hint">species guesses:
    {% for common, latin, p in result.top %}{{ common }} {{ (p * 100) | round(0) | int }}%{{ ", " if not loop.last else "" }}{% endfor %}
  </div>
</div>
{% endif %}

<h2>Species seen</h2>
{% if rows %}
<table>
  <tr><th>species</th><th>count</th><th>first seen</th><th>last seen</th><th>best photo</th></tr>
  {% for r in rows %}
  <tr>
    <td><a href="/?species={{ r.species | urlencode }}">{{ r.species }}</a></td>
    <td>{{ r.count }}</td>
    <td>{{ r.first_seen | local_time }}</td>
    <td>{{ r.last_seen | local_time }}</td>
    <td><a href="/photos/{{ r.best_path }}">{% if r.best_thumb_path %}<img src="/thumbs/{{ r.best_thumb_path }}" alt="">{% else %}photo{% endif %}</a></td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="hint">No species yet — confident species appear here as birds are captured.</p>
{% endif %}
</body>
</html>
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_web.py -q` → all pass (27 existing incl. Task 2's + 10 new). Full suite green.

- [ ] **Step 6: Commit**

```bash
git add birdy/web/ tests/test_web.py
git commit -m "Web: species filter, /species summary with photo upload, last_species"
```

---

### Task 7: Wire `create_app(classifier=)`, docs, deploy, verify

**Files:**
- Modify: `birdy/cli.py` (pass the classifier to `create_app`), `CLAUDE.md`, `deploy/deploy.sh` (nothing — assets are rsynced), `docs/superpowers/specs/2026-10-03-species-recognition-design.md` (only if code deviated)

- [ ] **Step 1: `cmd_run`**: `clf = _classifier(config, live)`; `pipeline = Pipeline(store, clf, live)`; `app = create_app(store, state, live, config_path, queue_depth=pipeline.queue_depth, classifier=clf)`. Note: the pipeline worker thread and the web thread then share one `TwoStageClassifier` — the TFLite interpreters are **not** thread-safe. Guard: give `TwoStageClassifier` a `self._lock = threading.Lock()` and wrap the two model calls in `classify_image`:
```python
        with self._lock:  # TFLite interpreters are not thread-safe; pipeline and web share this object
            p_bird = self.gate.predict(img_rgb)
            try:
                top = self.species.top(img_rgb, k=5)
            except Exception as e:
                ...  # unchanged dedup logging
```
Add to `tests/test_two_stage.py`:
```python
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
```

- [ ] **Step 2: `CLAUDE.md`**: architecture bullets — the classifier is now `TwoStageClassifier` (gate + `InatBirdClassifier`), rules for `species`/`rescue`; `species_checked` + backlog; `/species` and `POST /classify`; gotcha: the iNat model's output is already softmax-scaled (don't apply softmax), while the ImageNet gate's is logits; the two interpreters are shared by the pipeline and web threads under a lock. Commands: `birdy classify` prints both models.

- [ ] **Step 3: Tests + commit**: `.venv/bin/pytest -q` green, pyflakes silent. `git add -A birdy CLAUDE.md tests && git commit -m "Share the two-stage classifier with the upload box; document species recognition"`.

- [ ] **Step 4: Deploy + verify on the Pi**

`deploy/deploy.sh`. Then:
1. `ssh yassine@birdy.local 'journalctl -u birdy -n 20 --no-pager -o cat | grep -E "backfill|ERROR|starting"'` — expect `birdy starting` and, within ~1 s, `backfill capture N: Eurasian Blue Tit 0.xx` lines as the sweep drains today's phone-screen captures (50 per tick).
2. `curl -s http://birdy.local:8000/status` → `last_species` non-null once backfill reached a blue tit.
3. `curl -s http://birdy.local:8000/species | grep -c "Eurasian Blue Tit"` ≥ 1.
4. Upload test: `curl -s -o /dev/null -w "%{http_code}\n" -F "image=@tests/fixtures/blue_tit.jpg" http://birdy.local:8000/classify` → 200; and from the phone, open `/species` → Try a photo → pick a bird picture → result shown.
5. `ssh yassine@birdy.local 'cd ~/pi-birdy && time .venv/bin/birdy classify /tmp/bird.jpg'` → species section printed, < 3 s.
6. Watch CPU during backfill: `ssh yassine@birdy.local 'top -bn1 | head -12'` — the backlog is ~900 rows ≈ 18 sweeps ≈ 90 min at 5-min ticks; acceptable.

- [ ] **Step 5: Finish** → `superpowers:finishing-a-development-branch`.
