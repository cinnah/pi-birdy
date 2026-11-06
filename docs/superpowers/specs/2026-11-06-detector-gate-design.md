# Detector gate — design

Date: 2026-11-06. Builds on the capture core and species recognition (2) specs and on Part 1 (motion crop) of `2026-10-05-recognition-improvements-design.md`. A follow-up spec will replace the species model (BioCLIP 2); this one only decides *bird or not*.

## Why

Measured on 2026-11-06 over every capture on the Pi (641 rows, each crop looked at by hand): 117 photos show a bird, and the ImageNet gate at `classifier.threshold = 0.5` keeps 53 of them. The scene is a terracotta pot; for a dark passerine on its rim the gate answers "pot" at 0.8–0.95 and P(bird) lands at 0.0–0.45 on sharp, unobstructed birds. Species rescue is disabled on the Pi (`rescue_threshold = 1.0`) for a good reason: all 72 manual "not a bird" corrections were rescues where the iNat model saw "Hooded Oriole" or "New Zealand Pigeon" ≥ 0.7 in sunlit yellow leaves. The iNat model cannot gate either: its `background` output is 0 on every frame and its species mass is ~0.8 with or without a bird.

A COCO object detector run on the same crop separates the two populations. On the labelled set, with the ImageNet gate kept at 0.5 and OR-ed with the detector:

| Detector (Coral, Apache-2.0) | Birds kept of 117 | False positives | Pi 4, per crop |
|---|---|---|---|
| EfficientDet-Lite0, bird ≥ 0.2 | 104 | 10 (8 are one gardening session with a person in frame) | 168 ms |
| SSD-MobileNet v2, bird ≥ 0.2 | 95 | 7 | 143 ms |
| both | 108 | 13 | 311 ms |

The remaining misses are motion-blurred take-offs. Running the species model on the detector's tight box instead of the pipeline crop changes nothing (Black Redstart mean p 0.071 → 0.075 on 97 photos), so the box is stored but not used for species.

**Decision:** one extra model, EfficientDet-Lite0 (`efficientdet_lite0_320_ptq.tflite` from google-coral/test_data, 4.5 MB, uint8 320×320, post-processed outputs), OR-ed with the existing gate. Rescue stays as it is.

## Scope

In: the detector wrapper, the combined rule, stored detector columns, a backfill that re-gates stored non-birds, the crop view's box overlay, `birdy classify` output, a live `detector.threshold`, model files + docs + deploy. Out: using the detector box for the species crop (measured: no gain), person suppression (rare, handled by **Not a bird**), running the detector on the lores stream (it replaces nothing in motion detection), any change to species, alerts, publishing or retention rules.

## Detector wrapper — `birdy/detector.py`

```python
@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]   # (x, y, w, h) as fractions of the input image, clamped to [0, 1]

class BirdDetector:
    def __init__(self, model_path: Path, labels_path: Path, bird_label: str = "bird") -> None: ...
    def detect(self, img_rgb: np.ndarray) -> list[Detection]: ...      # all detections the model returned, best first
    def best_bird(self, img_rgb: np.ndarray) -> Detection | None: ...  # highest-scoring "bird", None if none
```

- Loads through `birdy.tflite.load_interpreter` like the other two models. Input: the RGB crop resized to the model's input size with `INTER_AREA`, fed as uint8 (the model is uint8; a float input dtype gets the same `/127.5 − 1` scaling `InatBirdClassifier` uses). Resizing a non-square crop squashes it; the ROI-fallback crop is 1.2:1 and the detector was measured on exactly these crops, so no letterboxing.
- Outputs are the four post-processed tensors (boxes `[1,N,4]` as `ymin,xmin,ymax,xmax` fractions, classes `[1,N]`, scores `[1,N]`, count `[1]`). They are matched **by shape**, not by index: boxes = the 3-D tensor, count = the 1-D one, and of the two `[1,N]` tensors the one whose values are all integral is `classes` (the other post-processed Coral model, SSD, orders them differently; this keeps the wrapper model-agnostic). Boxes are converted to `(x, y, w, h)` and clamped.
- Labels: `models/coco_labels.txt`, one name per line, **line index = class id** (`???` placeholders for the COCO ids the model doesn't use; `bird` is line 16, id 15). The constructor fails with `ValueError` if `bird_label` is not in the file.
- Not thread-safe (TFLite); only ever called under `TwoStageClassifier._lock`.

## The rule — `birdy/classifier.py`

`TwoStageClassifier.__init__` gains `detector: BirdDetector`. `classify_image` runs the gate, the detector and the species model on the **same** RGB crop, all under the one lock, and applies:

```
p_gate   = gate.predict(img)                      # P(bird), ImageNet
det      = detector.best_bird(img)                # None or Detection
p_det    = det.score if det else 0.0
label, p_sp = species top-1 (unchanged; species model failure handled as today)

species  = label.common if p_sp >= species.threshold else None       # unchanged
rescued  = p_gate < classifier.threshold and p_det < detector.threshold and p_sp >= species.rescue_threshold
is_bird  = p_gate >= classifier.threshold or p_det >= detector.threshold or rescued
confidence = max(p_gate, p_det, p_sp) if rescued else max(p_gate, p_det)
```

So `confidence` keeps its meaning ("how sure the system is this is a bird") and now reflects whichever signal said so. Publishing ranks on `species_confidence` and is unaffected; the grid badge and `/status` show the new value. A detector failure (exception) is logged once per distinct message and treated as `p_det = 0`, like a species failure — the gate keeps working.

`Prediction` gains `detector_confidence: float` (0.0 when no bird detection) and `detector_box: tuple[float, float, float, float] | None`. `classify_image` keeps returning `(pred, top)`; `cmd_classify` and the web upload box read the detector fields from `pred`.

Thresholds: `detector.threshold` is read from `live.get()` on every call, like the others. Rescue is unchanged in meaning and keeps its validation (`>= species.threshold`).

## Config

New section, `config.py` dataclass + validation, documented in `birdy.toml.example`:

```toml
[detector]
model = "models/efficientdet_lite0_320_ptq.tflite"   # restart-only
labels = "models/coco_labels.txt"                     # restart-only
threshold = 0.2            # a COCO "bird" detection at or above this makes the photo a bird (live)
```

`threshold` is validated to `[0, 1]`. Paths default like the other models' (`_path`), so the Pi's existing `birdy.toml` needs no edit. `detector.model/labels` join the restart-only list in the watcher's warning; `threshold` is live and on the settings page (`FIELD_KEYS["detector_threshold"] = "detector.threshold"`, `SettingsPatch.detector_threshold`, `config_to_dict`, `apply_patch`, `parse_settings_form`, a `num(...)` row under **Bird check** between bird certainty and species certainty, labelled "Bird detector", help "A second bird check that looks for a bird shape. A photo is a bird when either check passes."). `test_config_to_dict_covers_every_field` covers the new keys.

## Store

Nullable columns added through the existing `ALTER TABLE` loop (same duplicate-column tolerance):

| Column | Meaning |
|---|---|
| `det_confidence REAL` | best "bird" detection score at classification time; 0 when none; NULL = not yet run |
| `det_x, det_y, det_w, det_h REAL` | that detection's box as fractions of the **classified crop** (not the frame); NULL when none |
| `detector_checked INTEGER NOT NULL DEFAULT 0` | the row has been through the detector (live or backfill) |

`Capture` gains `det_confidence: float | None` and `det_box: tuple[float, float, float, float] | None`. `update_classification(capture_id, is_bird, confidence, species, species_confidence, det_confidence, det_box)` writes them and sets `detector_checked = 1` together with `species_checked = 1`. `mark_not_bird`'s copy of the model verdict into `model_*` is unchanged (`model_confidence` gets the combined `confidence`).

`species_backlog` / `count_species_backlog` become `classification_backlog` / `count_classification_backlog` with `WHERE status = 'done' AND (species_checked = 0 OR detector_checked = 0) AND manual_at IS NULL`, same `ORDER BY`/`LIMIT`. `mark_species_checked` becomes `mark_checked` and sets both flags.

## Pipeline

`Pipeline.process` and `_backfill_species` (renamed `_backfill`) pass `pred.detector_confidence` / `pred.detector_box` to `update_classification`. The backfill rules are unchanged and now also apply to the detector: never demote a bird (`is_bird = c.is_bird or pred.is_bird`), confidence from the new verdict when it says bird, a failure marks the row checked instead of flipping it to `error`, bounded to 50 rows per sweep, yields to live events, one summary line per sweep (`backfill: %d rows, %d with species, %d promoted, %d remaining`; "promoted" = was non-bird, now bird). So after the deploy every stored row is re-gated within `ceil(rows / 50)` sweeps (585 non-birds → about an hour) with no new command. Promoted rows get a species as today and go through `publisher.evaluate` when they have one; they never alert (backfill never does).

Per-capture cost on the Pi rises from ~0.15 s to ~0.35 s; the queue (50) and the 3 s cooldown make that invisible. A backfill sweep holds the classifier lock for up to ~20 s in 50 slices; the web upload box waits at most one slice.

## Web and CLI

- `GET /captures/{id}/crop.jpg` draws the stored `det_box` on the crop as a 2 px rectangle (`#ffcc00`) when present, so **Model view** shows what the detector locked onto. Captures without a box (none found, or rows from before this change until the backfill reaches them) render as today.
- The grid tile badge is unchanged (it shows `confidence`, which now carries the detector's say).
- `birdy classify <file>` / `--capture <id>` print a third block after the gate's top-5:
  ```
  detector (COCO): bird 0.62 box x=0.31 y=0.22 w=0.38 h=0.41   # or "detector (COCO): no bird (best: potted plant 0.55)"
  ```
  and the verdict line becomes `bird=True p_bird=0.62 (gate 0.11, detector 0.62) species=-`, with `, rescued by species` appended as today.
- `POST /classify` (species page upload box) shows the detector line in its result.

## Models, deploy, docs

- `models/download.sh` fetches `efficientdet_lite0_320_ptq.tflite` and `coco_labels.txt` from `https://raw.githubusercontent.com/google-coral/test_data/master/`; both files are committed (4.5 MB + 1 KB) and `models/README.md` documents them (Apache-2.0, same source as the species model; uint8 320×320; outputs post-processed; label line index = class id).
- `deploy/deploy.sh` already syncs `models/`; `install.sh` needs nothing new (same runtime).
- `CLAUDE.md`: the architecture bullet for the classifier (three models on one crop, the OR rule, `confidence` = max), the backfill bullet (now re-gates too, `detector_checked`), the config gotcha list (`detector.threshold` live, `model/labels` restart-only), and the tunable-key checklist (`detector.threshold` as the worked example is fine).
- The Pi's `rescue_threshold = 1.0` stays; nothing in this change makes rescue safer.

## Testing (all on the Mac)

- **Detector wrapper** (`tests/test_detector.py`): real model on `tests/fixtures/bird.jpg` → a `bird` detection with score ≥ 0.3 and a box inside `[0, 1]`; on `cat.jpg` → `best_bird` is None or < 0.2; a fake interpreter with the tensors in SSD order (count, boxes, classes, scores) and in EfficientDet order both decode to the same detections (shape matching); `ValueError` when the label file lacks `bird`; a float-input model gets scaled input.
- **Rule** (`tests/test_two_stage.py`, `FakeDetector(score, box)` beside the existing fakes): gate below / detector above → bird, `confidence` = detector score, `detector_box` set; gate above / detector below → bird, `confidence` = gate; both below → not a bird, `confidence` = max of the two; both below + species ≥ rescue → rescued, `confidence` = species; detector raising → gate-only verdict, logged once; `detector.threshold` read live (change `live`, re-classify).
- **Config / settings**: defaults, validation of `threshold`, round trip through the settings form, `config_to_dict` coverage, watcher warns on `detector.model` change.
- **Store**: migration adds the five columns and the `detector_checked` flag to an old-format DB (idempotent; a second open is a no-op); `update_classification` writes the det columns and both `checked` flags; `classification_backlog` returns rows with either flag 0 and skips corrected rows; `mark_checked` sets both.
- **Pipeline**: live `process` stores det fields; backfill promotes a stored non-bird the fake detector finds, never demotes, marks checked on failure, counts "promoted" in the summary line, calls `publisher.evaluate` for a promoted row with a species, never calls the notifier.
- **Web/CLI**: `crop.jpg` with a stored box contains the overlay colour, without one is unchanged; `birdy classify` prints the detector block (fake classifier); `/classify` result shows it.
