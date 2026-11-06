# Detector Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A COCO object detector (EfficientDet-Lite0) becomes a second bird/non-bird signal OR-ed with the ImageNet gate, so the ~45 % of bird photos the gate rejects land in the birds view; stored non-birds are re-gated by the existing backfill sweep.

**Architecture:** `birdy/detector.py` wraps the Coral TFLite detector (post-processed outputs matched by shape). `TwoStageClassifier` runs gate + detector + species on the same RGB crop under its lock and applies `bird = gate ≥ t_gate or detector ≥ t_det or rescued`, `confidence = max(...)`. The store gains `det_confidence`, `det_x/y/w/h`, `detector_checked`; the backfill query becomes "species_checked = 0 OR detector_checked = 0" so history is re-gated 50 rows per sweep. The crop view draws the stored box; `birdy classify` prints the detector block; `detector.threshold` is a live settings-page key.

**Tech Stack:** Python 3.11 (Mac) / 3.13 (Pi), ai-edge-litert, OpenCV, numpy, FastAPI + Jinja, SQLite, pytest. Spec: `docs/superpowers/specs/2026-11-06-detector-gate-design.md`.

**Conventions:** run tests with `.venv/bin/pytest`. Models are committed in `models/`. Every config key goes in `config.py` + `birdy.toml.example`. Commit after every task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` as the last line.

---

## File map

| File | Change |
|---|---|
| `models/download.sh`, `models/README.md`, `models/LICENSE`, `models/efficientdet_lite0_320_ptq.tflite`, `models/coco_labels.txt` | fetch + document the detector |
| `birdy/detector.py` (new) | `Detection`, `BirdDetector` |
| `birdy/config.py`, `birdy.toml.example` | `DetectorConfig` + `[detector]` |
| `birdy/watcher.py` | detector model/labels restart-only |
| `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` (settings values) | live `detector.threshold` |
| `birdy/classifier.py` | `Prediction` fields, `TwoStageClassifier(detector=...)`, the rule |
| `birdy/store.py` | columns, `Capture.det_*`, `update_classification`, `classification_backlog`, `mark_checked` |
| `birdy/pipeline.py` | det fields stored, `_backfill`, `BACKLOG_PER_SWEEP`, summary line |
| `birdy/cli.py` | `_classifier` builds the detector; `cmd_classify` prints it |
| `birdy/web/app.py`, `birdy/web/templates/species.html` | crop overlay; upload result line |
| `CLAUDE.md` | architecture + gotchas |
| tests: `tests/test_detector.py` (new), `test_config.py`, `test_watcher.py`, `test_settings.py`, `test_web.py`, `test_two_stage.py`, `test_store.py`, `test_pipeline.py`, `test_cli.py` | |

---

### Task 1: Model files

**Files:**
- Modify: `models/download.sh`, `models/README.md`, `models/LICENSE`
- Create (downloaded): `models/efficientdet_lite0_320_ptq.tflite`, `models/coco_labels.txt`

- [ ] **Step 1: Extend the download script**

Append to `models/download.sh` before the two `echo` lines:

```bash
curl -sSL -o efficientdet_lite0_320_ptq.tflite "$coral/efficientdet_lite0_320_ptq.tflite"
curl -sSL -o coco_labels.txt "$coral/coco_labels.txt"
echo "detector: $(du -h efficientdet_lite0_320_ptq.tflite | cut -f1), labels: $(wc -l < coco_labels.txt) lines"
```

- [ ] **Step 2: Fetch the two files**

Run: `cd models && ./download.sh && cd ..`
Expected: the last three lines report `detector: 4.4M, labels: 90 lines` (sizes may differ slightly), and `git status` shows the two new files plus no change to the existing models (`git diff --stat` empty for them).

- [ ] **Step 3: Verify the label layout**

Run: `sed -n '16p' models/coco_labels.txt`
Expected: `bird` (line 16 = class id 15; the file has one name per line and `???` placeholders).

- [ ] **Step 4: Document**

Append to `models/README.md`:

```markdown

`efficientdet_lite0_320_ptq.tflite` + `coco_labels.txt` are the Coral COCO object
detector (EfficientDet-Lite0, uint8 320×320, outputs post-processed: boxes as
`ymin,xmin,ymax,xmax` fractions, classes, scores, count) from
github.com/google-coral/test_data, Apache-2.0. Used as the second bird/not-bird
signal: a `bird` detection (label line 16, class id 15 — line index = id) at or
above `detector.threshold` makes a photo a bird even when the ImageNet gate says no.
```

Append to `models/LICENSE`:

```
efficientdet_lite0_320_ptq.tflite and coco_labels.txt: Coral test data
(github.com/google-coral/test_data), Apache License 2.0 -
https://www.apache.org/licenses/LICENSE-2.0
```

- [ ] **Step 5: Commit**

```bash
git add models/
git commit -m "Models: Coral EfficientDet-Lite0 COCO detector + labels (Apache-2.0)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Detector wrapper

**Files:**
- Create: `birdy/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detector.py`:

```python
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


@pytest.mark.skipif(not MODEL.exists() or not (FIXTURES / "bird.jpg").exists(), reason="model or fixture missing")
class TestRealModel:
    @pytest.fixture(scope="class")
    def det(self):
        return BirdDetector(MODEL, LABELS)

    def test_bird_fixture_has_a_bird(self, det):
        img = cv2.cvtColor(cv2.imread(str(FIXTURES / "bird.jpg")), cv2.COLOR_BGR2RGB)
        b = det.best_bird(img)
        assert b is not None and b.score >= 0.3
        assert all(0.0 <= v <= 1.0 for v in b.box)

    def test_cat_fixture_has_no_confident_bird(self, det):
        img = cv2.cvtColor(cv2.imread(str(FIXTURES / "cat.jpg")), cv2.COLOR_BGR2RGB)
        b = det.best_bird(img)
        assert b is None or b.score < 0.2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_detector.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'birdy.detector'`.

- [ ] **Step 3: Implement `birdy/detector.py`**

```python
"""COCO object detection (Coral EfficientDet-Lite0 / SSD-MobileNet TFLite) used as a second
bird/not-bird signal. The model's own post-processing gives boxes, classes, scores and a count."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from birdy.tflite import load_interpreter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # (x, y, w, h) as fractions of the input image, clamped to [0, 1]


class BirdDetector:
    """Not thread-safe (TFLite); callers serialise through `TwoStageClassifier._lock`."""

    def __init__(self, model_path: Path, labels_path: Path, bird_label: str = "bird") -> None:
        # One name per line; line index = class id ("???" marks unused COCO ids).
        self.labels = [l.strip() for l in Path(labels_path).read_text().splitlines()]
        if bird_label not in self.labels:
            raise ValueError(f"{labels_path}: no '{bird_label}' label")
        self.bird_label = bird_label
        self._interp = load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self._outputs = self._interp.get_output_details()

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        """Every detection the model returned, best first."""
        img = cv2.resize(img_rgb, (int(self._in_w), int(self._in_h)), interpolation=cv2.INTER_AREA)
        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 255.0)
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        tensors = [np.asarray(self._interp.get_tensor(d["index"])) for d in self._outputs]
        # Match outputs by shape, not position: Coral's SSD and EfficientDet order them differently.
        boxes = next(t[0] for t in tensors if t.ndim == 3)
        count = int(next(t for t in tensors if t.ndim == 1)[0])
        vecs = [t[0] for t in tensors if t.ndim == 2]
        a, b = vecs[0], vecs[1]
        if _integral(a) and not _integral(b):
            classes, scores = a, b
        elif _integral(b) and not _integral(a):
            classes, scores = b, a
        else:  # both integral (all zeros) or neither: scores are the ones within [0, 1]
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
```

Note on the float branch: the spec says "the same scaling `InatBirdClassifier` uses" (`/127.5 − 1`); Coral's float detectors take `[0, 1]` inputs instead, and the test pins `max == 1.0`. Keep `/255.0` and this note; no shipped model is float anyway.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_detector.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/detector.py tests/test_detector.py
git commit -m "Detector: BirdDetector wraps the Coral COCO model, outputs matched by shape

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Config section `[detector]`

**Files:**
- Modify: `birdy/config.py` (dataclass after `SpeciesConfig`; parse after the species block; `Config` field; constructor call), `birdy.toml.example` (after `[species]`), `birdy/watcher.py:16-26`
- Test: `tests/test_config.py`, `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_detector_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.detector.model == tmp_path / "models" / "efficientdet_lite0_320_ptq.tflite"
    assert cfg.detector.labels == tmp_path / "models" / "coco_labels.txt"
    assert cfg.detector.threshold == 0.2


def test_detector_threshold_range(tmp_path):
    with pytest.raises(ConfigError, match="detector.threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[detector]\nthreshold = 1.5\n"))
```

Append to `tests/test_watcher.py`:

```python
def test_detector_model_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[detector]\nmodel = "models/other.tflite"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert live.get().detector.model.name == "other.tflite"
    assert "restart birdy to apply them" in caplog.text
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py -k detector tests/test_watcher.py -k detector -q`
Expected: `AttributeError: 'Config' object has no attribute 'detector'` (3 failures).

- [ ] **Step 3: Implement**

`birdy/config.py` — after `SpeciesConfig`:

```python
@dataclass(frozen=True)
class DetectorConfig:
    model: Path               # restart-only
    labels: Path              # restart-only
    threshold: float          # a COCO "bird" detection at or above this makes the photo a bird (live)
```

In `Config`, after `species: SpeciesConfig`: `detector: DetectorConfig`.

In the parse function, after the rescue-threshold checks:

```python
    det = raw.get("detector", {})
    detector_threshold = _num(det, "detector", "threshold", 0.2, float)
    _check(0.0 <= detector_threshold <= 1.0, "detector.threshold must be between 0 and 1")
```

In the `Config(...)` constructor call, after the `species=SpeciesConfig(...)` block:

```python
        detector=DetectorConfig(
            model=_path(base, det, "detector", "model", "models/efficientdet_lite0_320_ptq.tflite"),
            labels=_path(base, det, "detector", "labels", "models/coco_labels.txt"),
            threshold=detector_threshold,
        ),
```

`birdy.toml.example` — after the `[species]` block:

```toml
[detector]
model = "models/efficientdet_lite0_320_ptq.tflite"   # COCO object detector; restart-only
labels = "models/coco_labels.txt"                     # one name per line, line index = class id; restart-only
threshold = 0.2            # a "bird" detection at or above this makes the photo a bird, whatever the gate said
```

`birdy/watcher.py` `_restart_only`: add `cfg.detector.model, cfg.detector.labels,` after the `cfg.species.model, cfg.species.labels,` line.

`birdy/settings.py` `config_to_dict`: after the `"species": {...}` entry add (so `test_config_to_dict_covers_every_field` stays green):

```python
        "detector": {
            "model": str(config.detector.model),
            "labels": str(config.detector.labels),
            "threshold": config.detector.threshold,
        },
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_config.py tests/test_watcher.py tests/test_settings.py -q`
Expected: all pass. If `test_example_config_loads` fails, the example TOML has a typo.

- [ ] **Step 5: Commit**

```bash
git add birdy/config.py birdy.toml.example birdy/watcher.py birdy/settings.py tests/test_config.py tests/test_watcher.py
git commit -m "Config: [detector] model/labels (restart-only) and threshold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Live `detector.threshold` on the settings page

**Files:**
- Modify: `birdy/settings.py` (`FIELD_KEYS`, `SettingsPatch`, `config_to_dict`, `apply_patch`, `parse_settings_form`), `birdy/web/templates/settings.html:251-256`, `birdy/web/app.py` (settings values dict, ~line 236)
- Test: `tests/test_settings.py`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_settings.py`, `default_patch`: add `detector_threshold=config.detector.threshold,` after `species_rescue_threshold=...`. In the `FORM` dict add `"detector_threshold": "0.2",` after `"species_rescue_threshold": "0.7",`. Append:

```python
def test_detector_threshold_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, detector_threshold=0.35))
    assert new.detector.threshold == 0.35
    assert load_config(config_path).detector.threshold == 0.35


def test_parse_form_detector_threshold():
    assert parse_settings_form(FORM).detector_threshold == 0.2
```

In `tests/test_web.py`, the settings form helper (line ~184): add `"detector_threshold": "0.2",` after `"species_rescue_threshold": "0.7",` in every form dict that lists it (lines ~186, ~289, ~306 — grep `species_rescue_threshold`). Append:

```python
def test_settings_page_has_detector_threshold(client):
    r = client.get("/settings").text
    assert 'name="detector_threshold"' in r and 'value="0.2"' in r
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_settings.py tests/test_web.py -k "detector or covers_every_field" -q`
Expected: `TypeError: SettingsPatch.__init__() got an unexpected keyword argument 'detector_threshold'`, the coverage test fails on `detector.model`, the page test fails.

- [ ] **Step 3: Implement**

`birdy/settings.py`:
- `FIELD_KEYS`: add `"detector_threshold": "detector.threshold",` after the rescue line.
- `SettingsPatch`: add `detector_threshold: float` after `species_rescue_threshold: float`.
- `config_to_dict`: already has the `"detector"` entry from Task 3.
- `apply_patch`: after the rescue line add `data["detector"]["threshold"] = patch.detector_threshold`.
- `parse_settings_form`: add `detector_threshold=_num(form, "detector_threshold", float),` after the rescue line.

`birdy/web/app.py` settings values dict: add `"detector_threshold": cfg.detector.threshold,` after `"species_rescue_threshold": ...`.

`birdy/web/templates/settings.html`, between the bird-certainty and species-certainty rows:

```jinja
    {{ num("detector_threshold", "detector.threshold", "Bird detector",
           "A second bird check that looks for a bird shape. A photo is a bird when either check passes.", 0, 1, 0.01) }}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_settings.py tests/test_web.py tests/test_config.py -q`
Expected: all pass (including `test_config_to_dict_covers_every_field`).

- [ ] **Step 5: Commit**

```bash
git add birdy/settings.py birdy/web/app.py birdy/web/templates/settings.html tests/test_settings.py tests/test_web.py
git commit -m "Settings: detector.threshold is live and on the settings page

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The combined rule in `TwoStageClassifier`

**Files:**
- Modify: `birdy/classifier.py` (`Prediction`, `TwoStageClassifier.__init__`, `classify_image`, module docstring)
- Test: `tests/test_two_stage.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_two_stage.py` add after `FakeSpecies`:

```python
class FakeDetector:
    def __init__(self, score=0.0, raise_exc=None):
        self.score, self.raise_exc = score, raise_exc

    def best_bird(self, img):
        if self.raise_exc:
            raise self.raise_exc
        if self.score <= 0:
            return None
        return Detection("bird", self.score, (0.1, 0.2, 0.3, 0.4))
```

and the import `from birdy.detector import Detection`. Change `two_stage` to:

```python
def two_stage(live, p_bird, p_sp, raise_exc=None, p_det=0.0, det_exc=None):
    return TwoStageClassifier(
        FakeGate(p_bird), FakeSpecies(p_sp, raise_exc), live, lores_size=(320, 240),
        detector=FakeDetector(p_det, det_exc),
    )
```

Update `test_classify_image_is_serialised_across_threads` to pass `detector=FakeDetector(0.0)` to its direct `TwoStageClassifier(...)` call. Append:

```python
def test_detector_alone_makes_a_bird(live):
    pred, _ = two_stage(live, 0.1, 0.2, p_det=0.6).classify_image(IMG)
    assert pred.is_bird is True and pred.confidence == 0.6
    assert pred.detector_confidence == 0.6 and pred.detector_box == (0.1, 0.2, 0.3, 0.4)


def test_gate_alone_keeps_working(live):
    pred, _ = two_stage(live, 0.9, 0.2, p_det=0.1).classify_image(IMG)
    assert pred.is_bird is True and pred.confidence == 0.9
    assert pred.detector_confidence == 0.1


def test_both_below_is_not_a_bird(live):
    pred, _ = two_stage(live, 0.1, 0.2, p_det=0.15).classify_image(IMG)
    assert pred.is_bird is False and pred.confidence == 0.15
    assert pred.detector_box is not None   # stored even for non-birds (debugging)


def test_no_detection_gives_zero_and_no_box(live):
    pred, _ = two_stage(live, 0.1, 0.2).classify_image(IMG)
    assert pred.detector_confidence == 0.0 and pred.detector_box is None


def test_rescue_still_works_with_detector_below(live):
    pred, _ = two_stage(live, 0.2, 0.75, p_det=0.1).classify_image(IMG)
    assert pred.is_bird is True and pred.confidence == 0.75


def test_detector_threshold_follows_live_config(live, config):
    clf = two_stage(live, 0.1, 0.2, p_det=0.25)
    assert clf.classify_image(IMG)[0].is_bird is True
    live.replace(dataclasses.replace(config, detector=dataclasses.replace(config.detector, threshold=0.3)))
    assert clf.classify_image(IMG)[0].is_bird is False


def test_detector_failure_keeps_gate_result(live, caplog):
    clf = two_stage(live, 0.9, 0.0, det_exc=RuntimeError("boom"))
    with caplog.at_level(logging.ERROR):
        pred, _ = clf.classify_image(IMG)
        clf.classify_image(IMG)
    assert pred.is_bird and pred.detector_confidence == 0.0
    assert caplog.text.count("detector failed") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_two_stage.py -q`
Expected: `TypeError: TwoStageClassifier.__init__() got an unexpected keyword argument 'detector'`.

- [ ] **Step 3: Implement**

`birdy/classifier.py`:

Module docstring → `"""Three models on one crop: an ImageNet bird/not-bird gate, a COCO bird detector and an iNat species model. Gate or detector makes a bird; a confident species verdict can rescue a frame both rejected."""`

Add `from birdy.detector import BirdDetector` to the imports.

`Prediction`:

```python
@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    # How sure the system is this is a bird: max(gate P(bird), detector score), and the
    # species probability too when a confident species rescued a rejected frame; 0..1.
    confidence: float
    species: str | None = None
    species_confidence: float | None = None
    detector_confidence: float = 0.0   # best COCO "bird" detection score; 0 when none
    detector_box: tuple[float, float, float, float] | None = None   # (x, y, w, h) fractions of the crop
```

`TwoStageClassifier.__init__`: add parameter `detector: BirdDetector | None = None` at the end; store `self.detector = detector` and `self._detector_errors: set[str] = set()`. Docstring: add "A COCO `bird` detection at or above `detector.threshold` makes the photo a bird too."

`classify_image` body:

```python
        cfg = self._live.get()
        t_gate = cfg.classifier.threshold if threshold is None else threshold
        t_det = cfg.detector.threshold
        t_sp, t_rescue = cfg.species.threshold, cfg.species.rescue_threshold

        with self._lock:
            p_bird = self.gate.predict(img_rgb)
            det = None
            if self.detector is not None:
                try:
                    det = self.detector.best_bird(img_rgb)
                except Exception as e:
                    key = f"{type(e).__name__}: {e}"
                    if key not in self._detector_errors:
                        log.exception("detector failed (%s); continuing without it", key)
                        self._detector_errors.add(key)
            try:
                top = self.species.top(img_rgb, k=5)
            except Exception as e:
                key = f"{type(e).__name__}: {e}"
                if key not in self._species_errors:  # log each distinct failure once, not per capture
                    log.exception("species model failed (%s); continuing with the gate only", key)
                    self._species_errors.add(key)
                top = []
        label, p_sp = top[0] if top else (None, 0.0)
        p_det = det.score if det is not None else 0.0

        species = label.common if label is not None and p_sp >= t_sp else None
        rescued = p_bird < t_gate and p_det < t_det and p_sp >= t_rescue
        is_bird = p_bird >= t_gate or p_det >= t_det or rescued
        confidence = max(p_bird, p_det, p_sp) if rescued else max(p_bird, p_det)
        pred = Prediction(
            is_bird=is_bird,
            confidence=confidence,
            species=species,
            species_confidence=p_sp if species is not None else None,
            detector_confidence=p_det,
            detector_box=det.box if det is not None else None,
        )
        return pred, top
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_two_stage.py tests/test_web.py tests/test_cli.py -q`
Expected: all pass (`detector=None` keeps existing constructions working; `test_no_rescue_between_thresholds` still sees `confidence == 0.2` because `p_det` is 0).

- [ ] **Step 5: Commit**

```bash
git add birdy/classifier.py tests/test_two_stage.py
git commit -m "Classifier: gate OR detector makes a bird; Prediction carries the detection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Store columns and the combined backlog

**Files:**
- Modify: `birdy/store.py` (`Capture`, migration loop, `_row_to_capture`, `update_classification`, backlog methods)
- Test: `tests/test_store.py`, and the renamed calls in `tests/test_pipeline.py`

- [ ] **Step 1: Rename in tests**

Run (BSD sed on the Mac):

```bash
sed -i '' -e 's/species_backlog(/classification_backlog(/g' -e 's/count_species_backlog(/count_classification_backlog(/g' -e 's/mark_species_checked(/mark_checked(/g' -e 's/test_species_backlog_newest/test_classification_backlog_newest/' -e 's/test_mark_species_checked/test_mark_checked/' -e 's/test_count_species_backlog/test_count_classification_backlog/' -e 's/test_sweep_drains_species_backlog/test_sweep_drains_classification_backlog/' tests/test_store.py tests/test_pipeline.py
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_store.py`:

```python
def test_update_classification_stores_detector_fields(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.6, None, None, det_confidence=0.6, det_box=(0.1, 0.2, 0.3, 0.4))
    c = store.get(cid)
    assert c.det_confidence == 0.6 and c.det_box == (0.1, 0.2, 0.3, 0.4)
    assert c.species_checked is True and store.classification_backlog() == []
    store.update_classification(cid, False, 0.0, None, None)
    c = store.get(cid)
    assert c.det_confidence == 0.0 and c.det_box is None


def test_backlog_includes_rows_not_yet_through_the_detector(store):
    cid = add(store, T0)
    store.update_classification(cid, False, 0.1, None, None)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET detector_checked = 0")
    assert [c.id for c in store.classification_backlog()] == [cid]
    assert store.count_classification_backlog() == 1
    store.mark_checked(cid)
    assert store.classification_backlog() == []


def test_migration_adds_detector_columns_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending', species_checked INTEGER NOT NULL DEFAULT 1);"
        "INSERT INTO captures (captured_at, path, motion_score, is_bird, status) VALUES ('2026-10-03T09:00:00.000+00:00', 'a.jpg', 1, 0, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    c = s.get(1)
    assert c.det_confidence is None and c.det_box is None
    assert [x.id for x in s.classification_backlog()] == [1]   # detector_checked defaults to 0
    s.close()
    Store(data).close()
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_store.py -q`
Expected: `AttributeError: 'Store' object has no attribute 'classification_backlog'` and friends.

- [ ] **Step 4: Implement**

`birdy/store.py`:

`Capture` — add after `model_species_confidence`:

```python
    # Best COCO "bird" detection at classification time: score (0 = none, None = not run yet)
    # and its box as fractions (x, y, w, h) of the classified crop.
    det_confidence: float | None = None
    det_box: tuple[float, float, float, float] | None = None
```

Migration loop — add to the tuple after `("manual_at", "TEXT"),`:

```python
                ("det_confidence", "REAL"),
                ("det_x", "REAL"),
                ("det_y", "REAL"),
                ("det_w", "REAL"),
                ("det_h", "REAL"),
                ("detector_checked", "INTEGER NOT NULL DEFAULT 0"),
```

`_row_to_capture` — add:

```python
        det_confidence=row["det_confidence"],
        det_box=None if row["det_x"] is None else (row["det_x"], row["det_y"], row["det_w"], row["det_h"]),
```

`update_classification`:

```python
    def update_classification(
        self,
        capture_id: int,
        is_bird: bool,
        confidence: float,
        species: str | None,
        species_confidence: float | None,
        det_confidence: float = 0.0,
        det_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        bx = (None, None, None, None) if det_box is None else det_box
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ?,"
                " det_confidence = ?, det_x = ?, det_y = ?, det_w = ?, det_h = ?,"
                " species_checked = 1, detector_checked = 1, status = 'done' WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, det_confidence, *bx, capture_id),
            )
```

Backlog methods (replace the three):

```python
    def count_classification_backlog(self) -> int:
        with self._lock:
            return int(self._conn.execute(_BACKLOG_COUNT).fetchone()[0])

    def classification_backlog(self, limit: int = 50) -> list[Capture]:
        """Classified rows a newer model hasn't seen (species or detector), newest first.
        Corrected rows never: re-running the models must not undo a person's correction."""
        with self._lock:
            rows = self._conn.execute(_BACKLOG_SELECT, (limit,)).fetchall()
        return [_row_to_capture(r) for r in rows]

    def mark_checked(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET species_checked = 1, detector_checked = 1 WHERE id = ?", (capture_id,))
```

and module-level constants after `FILTERS`:

```python
_BACKLOG_WHERE = "status = 'done' AND (species_checked = 0 OR detector_checked = 0) AND manual_at IS NULL"
_BACKLOG_COUNT = f"SELECT COUNT(*) FROM captures WHERE {_BACKLOG_WHERE}"
_BACKLOG_SELECT = f"SELECT * FROM captures WHERE {_BACKLOG_WHERE} ORDER BY captured_at DESC, id DESC LIMIT ?"
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_store.py -q`
Expected: all pass. `tests/test_pipeline.py` still fails on `species_backlog` inside `birdy/pipeline.py` until Task 7.

- [ ] **Step 6: Commit**

```bash
git add birdy/store.py tests/test_store.py tests/test_pipeline.py
git commit -m "Store: detector columns, detector_checked, one classification backlog

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Pipeline stores the detection and re-gates history

**Files:**
- Modify: `birdy/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py` replace the `SPECIES_BACKLOG_PER_SWEEP` import/usage in `test_backlog_limit_per_sweep` with `BACKLOG_PER_SWEEP`. Append:

```python
class DetectorFake(FakeClassifier):
    """What the real classifier returns for a bird the gate missed but the detector found."""

    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
        super().classify(image_path, roi_bbox, threshold, pad)
        return Prediction(is_bird=True, confidence=0.55, detector_confidence=0.55, detector_box=(0.1, 0.2, 0.3, 0.4))


def test_process_stores_detector_fields(store, live):
    ev = add(store, 0)
    Pipeline(store, DetectorFake(), live).process(ev.capture_id, ev.path, ev.captured_at)
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.55
    assert c.det_confidence == 0.55 and c.det_box == (0.1, 0.2, 0.3, 0.4)


def detector_unchecked(store, ids):
    with store._lock, store._conn:
        store._conn.execute(f"UPDATE captures SET detector_checked = 0 WHERE id IN ({','.join('?' * len(ids))})", ids)


def test_backfill_promotes_non_bird_the_detector_finds(store, live, caplog):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    with caplog.at_level(logging.INFO):
        Pipeline(store, DetectorFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.55 and c.det_box == (0.1, 0.2, 0.3, 0.4)
    assert store.classification_backlog() == []
    assert "1 promoted" in caplog.text


def test_backfill_never_demotes_on_detector_pass(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    detector_unchecked(store, [ev.capture_id])
    Pipeline(store, NotBirdFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.9 and store.classification_backlog() == []


def test_backfill_failure_marks_detector_checked(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    Pipeline(store, FakeClassifier(fail_on={ev.path.name}), live).sweep()
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.is_bird is False and store.classification_backlog() == []
```

(`NotBirdFake` already exists in the file.) The existing alert test that asserts the backfill never notifies keeps passing unchanged.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: `ImportError: cannot import name 'BACKLOG_PER_SWEEP'` plus `AttributeError ... species_backlog` from the pipeline.

- [ ] **Step 3: Implement**

`birdy/pipeline.py`:

- `SPECIES_BACKLOG_PER_SWEEP = 50` → `BACKLOG_PER_SWEEP = 50`.
- In `process`, the store call becomes
  ```python
            self._store.update_classification(
                capture_id, pred.is_bird, pred.confidence, pred.species, pred.species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box,
            )
  ```
- In `sweep`, replace the backfill block with:
  ```python
        # Rows classified before species recognition or the detector existed: re-run the
        # classifier so they get a species and a second look at bird/not-bird. Bounded per
        # sweep to keep the Pi responsive, and yields to live events.
        processed = with_species = promoted = 0
        for c in self._store.classification_backlog(limit=BACKLOG_PER_SWEEP):
            if self._stop.is_set() or self._queue.qsize():
                break
            got_species, was_promoted = self._backfill(c)
            processed += 1
            with_species += got_species
            promoted += was_promoted
        if processed:
            remaining = self._store.count_classification_backlog()
            log.info("backfill: %d rows, %d with species, %d promoted, %d remaining", processed, with_species, promoted, remaining)
  ```
- Rename `_backfill_species` → `_backfill`; its docstring `"""Re-classify an already-classified row; returns (got_species, promoted)."""`; the store call becomes
  ```python
            self._store.update_classification(
                c.id, is_bird, confidence, pred.species, pred.species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box,
            )
  ```
  and the failure branch calls `self._store.mark_checked(c.id)` (log line: `"backfill failed for capture %d; marking checked"`, `"mark_checked failed for capture %d"`). The return stays `bool(pred.species), (not c.is_bird) and pred.is_bird`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_store.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/pipeline.py tests/test_pipeline.py
git commit -m "Pipeline: store the detection; the backfill sweep re-gates stored non-birds

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: CLI builds the detector and `birdy classify` prints it

**Files:**
- Modify: `birdy/cli.py` (`_classifier`, `cmd_classify`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, both inline configs (`test_classify_command_prints_prediction` and `_classify_config`) get a detector section appended to the written TOML string:

```python
        f'[detector]\nmodel = "{ROOT}/models/efficientdet_lite0_320_ptq.tflite"\n'
        f'labels = "{ROOT}/models/coco_labels.txt"\n'
```

In `test_classify_command_prints_prediction` add the assertions:

```python
    assert "detector (COCO):" in proc.stdout
    assert "(gate " in proc.stdout and ", detector " in proc.stdout
```

Append:

```python
def test_missing_detector_model_exits_2(tmp_path):
    cfg = _classify_config(tmp_path, f"{ROOT}/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
    cfg.write_text(cfg.read_text().replace("efficientdet_lite0_320_ptq.tflite", "nope.tflite"))
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "bird.jpg"))
    assert proc.returncode == 2 and "model error" in proc.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -k "classify or detector" -q`
Expected: `test_classify_command_prints_prediction` fails on `detector (COCO):`; `test_missing_detector_model_exits_2` fails (returncode 0, the detector isn't loaded yet).

- [ ] **Step 3: Implement**

`birdy/cli.py`:

Add `from birdy.detector import BirdDetector` to the imports. `_classifier`:

```python
def _classifier(config: Config, live: LiveConfig) -> TwoStageClassifier:
    try:
        gate = ImageNetBirdClassifier(
            config.classifier.model,
            config.classifier.labels,
            config.classifier.bird_classes,
            threshold=config.classifier.threshold,
            lores_size=config.camera.lores_size,
        )
        species = InatBirdClassifier(config.species.model, config.species.labels)
        detector = BirdDetector(config.detector.model, config.detector.labels)
    except (ValueError, OSError, RuntimeError) as e:  # LiteRT raises RuntimeError on corrupt models
        raise ModelLoadError(str(e)) from e
    return TwoStageClassifier(gate, species, live, lores_size=config.camera.lores_size, detector=detector)
```

`cmd_classify`, replace from `p_gate = clf.gate.predict(img)` to the end of the gate block with:

```python
    p_gate = clf.gate.predict(img)
    pred, top = clf.classify_image(img)
    threshold = config.classifier.threshold
    rescued = ", rescued by species" if pred.is_bird and p_gate < threshold and pred.detector_confidence < config.detector.threshold else ""
    print(
        f"bird={pred.is_bird} p_bird={pred.confidence:.3f} (gate {p_gate:.3f}, detector {pred.detector_confidence:.3f}{rescued}) "
        f"species={pred.species or '-'}"
    )
    print("gate (ImageNet):")
    for label, prob in clf.gate.top_k_image(img, k=5):
        print(f"  {prob:.3f}  {label}")
    dets = clf.detector.detect(img) if clf.detector is not None else []
    bird = next((d for d in dets if d.label == clf.detector.bird_label), None) if dets else None
    if bird is not None:
        x, y, w, h = bird.box
        print(f"detector (COCO): bird {bird.score:.2f} box x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f}")
    elif dets:
        print(f"detector (COCO): no bird (best: {dets[0].label} {dets[0].score:.2f})")
    else:
        print("detector (COCO): no detections")
```

(The species block that follows is unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: all pass. Then a manual check: `.venv/bin/birdy --config birdy.toml.example classify tests/fixtures/bird.jpg` (from the repo root, the example's relative model paths resolve) prints the three blocks and a `bird=True` verdict.

- [ ] **Step 5: Commit**

```bash
git add birdy/cli.py tests/test_cli.py
git commit -m "CLI: the service and birdy classify load the detector; classify prints its verdict

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Crop view overlay and upload-box result

**Files:**
- Modify: `birdy/web/app.py` (`_crop_jpeg`, `classify_upload` result), `birdy/web/templates/species.html:69-72`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web.py` (next to the other crop tests; `_add_photo` exists there):

```python
def test_crop_jpg_draws_the_detector_box(client, store):
    cid = _add_photo(store, (100 / 320, 100 / 240, 20 / 320, 20 / 240))   # 80x80 crop
    plain = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    store.update_classification(cid, True, 0.6, None, None, det_confidence=0.6, det_box=(0.25, 0.25, 0.5, 0.5))
    boxed = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    assert boxed.shape == plain.shape
    b, g, r = boxed[20, 40]   # top edge of the box, BGR; #ffcc00 = (0, 204, 255) ± JPEG
    assert r > 200 and g > 150 and b < 80
    assert not np.array_equal(plain[20, 40], boxed[20, 40])


def test_classify_upload_shows_detector(client, classifier, store):
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    r = client.post("/classify", files={"image": ("b.jpg", data, "image/jpeg")})
    assert r.status_code == 200 and "detector 42%" in r.text
```

and in `FakeTwoStage.classify_image` set `detector_confidence=0.42, detector_box=(0.1, 0.1, 0.5, 0.5)` on its `Prediction`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_web.py -k "detector" -q`
Expected: both fail (no overlay colour; no "detector 42%").

- [ ] **Step 3: Implement**

`birdy/web/app.py` `_crop_jpeg`:

```python
    def _crop_jpeg(c) -> bytes | None:
        """What the classifier saw: the motion crop (else the ROI crop), with the stored detector box drawn."""
        crop = crop_box(c.motion_box, live.get().motion.roi, lores_size)
        try:
            rgb = CropLoader(lores_size).load(store.photos_dir / c.path, crop.bbox, crop.pad)
        except ValueError:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if c.det_box is not None:
            h, w = bgr.shape[:2]
            x, y, bw, bh = c.det_box
            cv2.rectangle(bgr, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (0, 204, 255), 2)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else None
```

`classify_upload` result dict: add `"detector": pred.detector_confidence,`.

`birdy/web/templates/species.html`: change `<span>gate + species</span>` to `<span>gate + detector + species</span>` and after the `{% if result.species %}...{% endif %}` span add:

```jinja
        <span class="det">detector {{ (result.detector * 100) | round(0) | int }}%</span>
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/web/app.py birdy/web/templates/species.html tests/test_web.py
git commit -m "Web: model view draws the detector box; upload box shows the detector score

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Docs, full suite, deploy

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

In the Architecture section, replace the `birdy/classifier.py` bullet's first sentences with:

> `birdy/classifier.py` exposes a `Classifier` protocol returning `Prediction(is_bird, confidence, species, species_confidence, detector_confidence, detector_box)`. The implementation is `TwoStageClassifier`: an ImageNet gate (`ImageNetBirdClassifier.predict` → P(bird), summed over ImageNet bird classes from MobileNetV2), a COCO object detector (`birdy/detector.py`, Coral EfficientDet-Lite0; `BirdDetector.best_bird` → the top `bird` detection with its box) and `InatBirdClassifier` (`birdy/species.py`, 965 iNat bird species), all run on one decoded crop (`CropLoader`) under one lock. A photo is a bird when the gate ≥ `classifier.threshold` **or** the detector ≥ `detector.threshold`; `confidence` is the max of the two. Species is reported when its probability ≥ `species.threshold`; a frame both rejected is *rescued* (`is_bird=True`, `confidence = max(gate, detector, species)`) when species ≥ `species.rescue_threshold`. The detector box is stored (`det_confidence`, `det_x/y/w/h`, fractions of the classified crop) and drawn on the **Model view** crop; it is not used for the species crop (measured 2026-11-06: no gain).

Replace the `species_checked` bullet with:

> `species_checked` and `detector_checked` mark rows the species model / the detector have seen. Each pipeline sweep backfills up to 50 older `done` rows with either flag unset (`_backfill`, `store.classification_backlog`): it never demotes a bird to non-bird, promotes a non-bird the detector or a rescue finds, never flips a `done` row to `error` on failure (it marks it checked instead), yields as soon as a live event is queued, never alerts, and logs one summary line per sweep (`… N promoted …`). After a deploy that adds a model, history is re-gated within `rows / 50` sweeps (5 min each).

In Conventions, add after the `telegram.enabled, …` restart-only bullet:

> `detector.model/labels` are restart-only (the watcher warns); `detector.threshold` is live and on `/settings`. Default 0.2 was measured on 2026-11-06 (`docs/superpowers/specs/2026-11-06-detector-gate-design.md`): with it the system keeps 104 of 117 real birds instead of 53. Person-in-frame false positives are rare and handled with **Not a bird**.

Add `birdy classify` note to the gotcha that describes it: "…then the gate's top-5, the detector's verdict (`detector (COCO): bird 0.62 box …` or `no bird (best: potted plant 0.55)`), then the species top-5."

- [ ] **Step 2: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass, no warnings about the detector. If `test_example_config_loads` or `test_full_config_roundtrip` fail, the `[detector]` section in `birdy.toml.example` / `FULL_TOML` is missing a key.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Docs: detector gate in CLAUDE.md

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Deploy and verify on the Pi** (only when the user says to deploy)

```bash
deploy/deploy.sh
ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy classify --capture 2460 | head -12'
```

Expected: `bird=True p_bird=0.4xx (gate 0.419, detector 0.4xx) species=-` and a `detector (COCO): bird …` line (capture 2460 is a Black Redstart the gate scored 0.42). Then `journalctl -u birdy -f` shows `backfill: 50 rows, … promoted, … remaining` lines every 5 minutes until the backlog drains, and `/?filter=birds` fills up.
