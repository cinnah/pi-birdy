# Birdy — Species Recognition Design

**Date:** 2026-10-03
**Status:** approved for implementation
**Sub-project:** 2 (follows capture core 1 and settings page 1b)
**Depends on:** `master` at the settings-page merge (`b6a6279`).

## Goal

Say *which* bird visited. Every capture is run through a second, species-level model; confident species are stored and shown on the tiles, a species filter and a `/species` summary page answer "what visited?", and an upload box lets the user test both models with a phone photo.

## Decisions made

| Question | Decision |
|---|---|
| Species model | Coral iNat bird classifier `mobilenet_v2_1.0_224_inat_bird_quant.tflite` (MobileNetV2, 965 classes incl. `background`, uint8, output already softmax-scaled 1/256). Downloaded from `https://raw.githubusercontent.com/google-coral/test_data/master/` (model + `inat_bird_labels.txt`), Apache-2.0, checked into `models/`. Measured on the Mac: phone-screen blue tit → *Eurasian Blue Tit* 0.59; cat → 0.12; unknown (Australian) bird → 0.23. |
| Wrong-continent guesses | Threshold only: a species is shown when `species.threshold` (default 0.5) is met, else the capture is a plain "bird". No regional allowlist. |
| When the species model runs | On every capture (gate-accepted or not). A gate-rejected frame is **rescued** (promoted to `is_bird = 1`) only when the species confidence meets the higher `species.rescue_threshold` (default 0.7). |
| Web | Tile label (exists) + species dropdown filter on `/` + a `/species` summary page (count, first/last seen, best photo) + `last_species` in `/status`. |
| Testing with phone photos | Upload box on `/species` (`POST /classify`): runs both models on the uploaded image in memory, shows the result, stores nothing. |
| Wiring | A `TwoStageClassifier` implementing the existing `Classifier` protocol; the pipeline is unchanged. |

## Architecture

```
pipeline ──classify(path, bbox, threshold)──▶ TwoStageClassifier
                                                │ decode + crop once
                                     ┌──────────┴───────────┐
                              ImageNetBirdClassifier   InatBirdClassifier
                              .predict(img) → P(bird)  .predict(img) → [(label, p)…]
                                     └──────────┬───────────┘
                                                ▼
                     Prediction(is_bird, confidence, species, species_confidence)
```

### `birdy/species.py` — `InatBirdClassifier`
- `InatBirdClassifier(model_path, labels_path)`; loads the interpreter once (same `_load_interpreter` as the gate, moved to a tiny `birdy/tflite.py` so both modules share it).
- Labels: each line is `Latin name (Common Name)` or `background`. Parsed into `SpeciesLabel(latin, common)`; `background` has `latin="background"`, `common="background"`.
- `predict(img_rgb: np.ndarray) -> list[tuple[SpeciesLabel, float]]`: resize to the model input (224×224, INTER_AREA), run, dequantize with the output's `(scale, zero_point)`; **no softmax** (the output is already softmax-scaled at 1/256; uint8 quantisation truncates the long tail, so the probabilities sum to ~0.8–0.95, not ≈ 1 — never re-softmax); return all classes sorted by probability, descending.
- `top(img_rgb, k=5, skip_background=True)` convenience.
- Pure function `parse_label(line) -> SpeciesLabel` for testing.

### `birdy/classifier.py` changes
- `ImageNetBirdClassifier` gains `predict(img_rgb) -> float` (P(bird) for an already-decoded RGB image); `classify(path, bbox, threshold)` now decodes/crops (via a shared `load_crop(path, bbox, lores_size, pad)` helper that keeps the empty-crop-fallback behaviour) and calls `predict`. Behaviour and tests unchanged.
- New `TwoStageClassifier(gate: ImageNetBirdClassifier, species: InatBirdClassifier, live: LiveConfig)` implementing `Classifier`:
  1. `img = load_crop(path, bbox, …)`; `p_bird = gate.predict(img)`.
  2. `label, p_sp = species.top(img, k=1, skip_background=True)[0]`; on any exception from the species model: log once per distinct error text, treat as `p_sp = 0`, `label = None`.
  3. `cfg = live.get()`; `t_gate = threshold if threshold is not None else cfg.classifier.threshold`; `t_sp = cfg.species.threshold`; `t_rescue = cfg.species.rescue_threshold`.
  4. `species = label.common if p_sp >= t_sp else None`; `species_confidence = p_sp if species else None`.
  5. `is_bird = p_bird >= t_gate or p_sp >= t_rescue`.
  6. `confidence = p_bird`, except when rescued (`p_bird < t_gate and p_sp >= t_rescue`): `confidence = max(p_bird, p_sp)`.
  - Steps 2–6 live in `classify_image(img_rgb, threshold=None) -> tuple[Prediction, list[tuple[SpeciesLabel, float]]]` (returns the prediction plus the species top-5 for display); `classify(path, bbox, threshold)` = `load_crop` + `classify_image(...)[0]`. The upload box calls `classify_image` directly on the decoded upload.
- `cli._classifier(config)` builds the two-stage classifier; `birdy classify <file>` prints `bird=… p_bird=…` then the gate's top-5 ImageNet labels and the species top-5 (common names with %).

### Config (`[species]`)
```toml
[species]
model = "models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"
labels = "models/inat_bird_labels.txt"
threshold = 0.5           # show a species at or above this
rescue_threshold = 0.7    # promote a gate-rejected frame to "bird" at or above this
```
Validation: both thresholds in [0, 1]; `rescue_threshold >= threshold`. `threshold` and `rescue_threshold` are editable on the settings page (`SettingsPatch` fields `species_threshold`, `species_rescue_threshold`); paths are restart-only like the other model paths.

### Store
- Migration at startup: if `captures` lacks `species_checked`, `ALTER TABLE captures ADD COLUMN species_checked INTEGER NOT NULL DEFAULT 0`. Idempotent (checked via `PRAGMA table_info`). Index `captures_species ON captures(species, captured_at DESC)`.
- `update_classification(...)` sets `species_checked = 1` (every write now comes from the two-stage classifier).
- `species_backlog(limit=50) -> list[Capture]`: `status = 'done' AND species_checked = 0`, newest first.
- `mark_species_checked(id)`: used when a backlog row fails, so one bad row can't loop forever.
- `species_summary() -> list[SpeciesRow(species, count, first_seen, last_seen, best_capture_id, best_thumb_path)]`, ordered by `last_seen` desc.
- `list_captures(filter, limit, offset, species=None)`, `count(filter, species=None)`: extra `AND species = ?` when given.
- `last_species() -> tuple[str, datetime] | None`.

### Pipeline
- Unchanged for live captures.
- The existing 5-minute sweep (and the startup sweep) additionally drains `species_backlog(50)` through a dedicated `_backfill_species(capture)` (not `process`): it classifies the photo and writes through `update_classification`, which sets `species_checked = 1`.
- Backfill never demotes: a row already judged a bird stays `is_bird = 1` with its old confidence even if the gate now disagrees; a non-bird row that the species model rescues becomes `is_bird = 1` (and is then exempt from retention).
- On exception it calls `mark_species_checked(id)` + logs; a failure never flips a `done` row to `error`.
- It yields to live captures: between backlog rows it stops as soon as a live event is queued, so the backlog never delays a fresh capture; the rest waits for the next sweep.
- Each sweep that processed rows logs one summary line: `backfill: N rows, N with species, N rescued, N remaining`.

### Web
- `GET /`: species `<select>` next to the filter links, options from `species_summary()` ("All species" + each name with its count); `?species=<name>` filters; unknown name → 400 (same style as the filter check).
- `GET /species`: table (species, count, first seen, last seen, best photo thumb → full-res), empty-state text, and the **Try a photo** form (`<input type="file" name="image" accept="image/*">`, `capture` attribute for phones).
- `POST /classify`: multipart image ≤ 10 MB → `cv2.imdecode` → the two-stage classifier's `classify_image(img_rgb)` (a path-free entry point; no ROI crop) → re-render `/species` with a result block: bird/not-bird with %, species top-3 with %. Nothing is written to disk or DB. Errors: no file / unreadable → 422; > 10 MB → 413.
- `/status` adds `last_species` (name or null).
- Settings page: two rows under Classifier (`species threshold`, `species rescue threshold`), range 0–1 step 0.01.
- Header link: "species".

## Error handling

| Case | Behaviour |
|---|---|
| Species model file missing/unloadable at startup | `birdy run` exits 2 with a clear message (the service does not restart-loop) |
| Species inference fails on one image | Logged (deduplicated by message); gate result stored with `species = None`; row `done`, `species_checked = 1` |
| Backlog row fails | Logged, `species_checked = 1`, not retried |
| Upload not an image / unreadable | 422 with message; nothing stored |
| Upload > 10 MB | 413 |
| `rescue_threshold < threshold` in config or form | `ConfigError` naming `species.rescue_threshold` |

## Performance (Pi 4)
Per capture ≈ one JPEG decode (~150 ms) + gate (~60 ms) + species (~100 ms) ≈ 0.3 s, inside the 3 s cooldown. Backfill ≤ 50 rows per sweep tick.

## Testing (Mac)
- `species.py`: `parse_label` (normal, `background`, no parentheses, nested parentheses in the common name); real model: blue-tit fixture (the Pi's phone-screen capture, cropped, checked into `tests/fixtures/blue_tit.jpg`) → top common name "Eurasian Blue Tit"; `cat.jpg` → top prob < 0.5; probabilities within [0, 1], `0.5 ≤ sum ≤ 1.01` and `max > 0.5` with no re-softmax (quantisation truncates the tail, so the sum is ~0.8–0.95); `skip_background` works.
- `TwoStageClassifier` with fake gate/species (configurable returns): bird+species; bird+unknown (species `None`); rescue at ≥ 0.7 with `confidence = max`; no rescue at 0.6; species exception → gate result kept, `species None`; `threshold=` override still honoured for the gate.
- `classifier.py`: existing tests unchanged; `load_crop` fallback test moved from the classifier.
- Store: migration on a fresh DB and on a DB created by the previous schema (create the old table by hand, open `Store`, assert the column exists; open again → no error); `species_backlog`, `mark_species_checked`, `species_summary` (count/first/last/best), species filter + count, `last_species`.
- Pipeline: sweep drains a seeded backlog (rows `done` with `species_checked = 0`) and marks them; a failing row is marked checked after one attempt.
- Config/settings: `[species]` defaults, validation (`rescue_threshold >= threshold`), coverage test picks up the new keys, form rows round-trip.
- Web: dropdown lists species with counts; `?species=` filters and 400 on unknown; `/species` table, best-photo link, empty state; `POST /classify` with `bird.jpg` → 200, "bird" shown, a species list rendered; garbage → 422; oversize → 413; nothing written (DB count and photo dir unchanged); `/status.last_species`; settings page shows both new rows.
- CLI: `birdy classify` prints a species section.
- Pi: deploy; upload a phone photo via the box; journal shows the backlog draining; `/species` lists Eurasian Blue Tit from today's phone-screen captures.

## Out of scope
Regional allowlist, retraining, Telegram alerts (next sub-project, will key on `species`), audio (BirdNET), per-species statistics beyond the summary table, editing/correcting a species label by hand.
