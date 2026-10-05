# Recognition improvements — design

Date: 2026-10-05. Builds on the capture core, species recognition (2), Telegram alerts (6) and publishing (3) specs.

## Why

The species model sees the box around the whole ROI polygon (`Pipeline.process` → `roi_bbox(cfg.motion.roi)`). At the feeder's 1.5 m a tit is ~210 px wide in the 3280 px still, but after the ROI crop is squeezed to 224×224 it is ~25–40 px — too small to identify. Each capture is also judged alone (labels flicker within one visit), there is no way to correct a wrong label, and the 965-class model guesses species from other continents.

Four parts, built in this order; each is useful on its own:

1. **Motion crop** — classify the area that moved, not the whole ROI.
2. **Visits and voting** — group a bird's captures into a visit, decide one label per visit, alert once per visit.
3. **Corrections** — the user's labels, stored next to the model's, applied per visit.
4. **Per-site species list** — each Pi limits the model to species recorded near its own location.

Out of scope: learning from corrections (embeddings + kNN), replacing either model, audio. Part 3 keeps the data they will need (raw model output, marked photos kept).

## Data model

`captures` gains (all nullable; added by migration):

| Column | Meaning |
|---|---|
| `motion_x`, `motion_y`, `motion_w`, `motion_h` | Raw motion box at the trigger frame, lores coords. NULL = none (old rows, no box) |
| `model_is_bird`, `model_confidence`, `model_species`, `model_species_confidence` | Exactly what the classifier said for this photo |
| `model_top5` | JSON `[[common, latin, p], …]`: the species model's top 5 after the species-list mask (part 4) |
| `visit_id` | FK → `visits.id`; NULL for photos the model and the user both call non-bird |
| `manual_is_bird`, `manual_species`, `manual_at` | The user's correction on this photo (part 3); NULL = none |

`is_bird`, `confidence`, `species`, `species_confidence` keep their names and every existing reader (grid filters, species dropdown, `/species`, `last_species`, `rank_day`, manifest, retention) keeps reading them — but they now hold the **final** label, written only by `resolve_visit` (and by the classifier write for photos outside visits).

New table:

```sql
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,      -- UTC ISO, first photo
  ended_at TEXT NOT NULL,        -- UTC ISO, last photo
  closed INTEGER NOT NULL DEFAULT 0,
  species TEXT,                  -- resolved visit label (NULL = bird, no species)
  species_confidence REAL,
  alert TEXT NOT NULL DEFAULT 'pending'   -- pending | sent | skipped
);
CREATE INDEX IF NOT EXISTS visits_open ON visits(closed, ended_at);
CREATE INDEX IF NOT EXISTS captures_visit ON captures(visit_id);
```

Migration (on `Store` open, once, guarded by a `PRAGMA user_version` bump): add columns; copy `is_bird/confidence/species/species_confidence` into `model_*`; build visits over all history (`assign_visit` in `captured_at` order, every visit `closed = 1`, `alert = 'skipped'`); resolve every visit; the publisher then re-ranks all history as `--rerank` does. Old rows have `model_top5` NULL and vote with `model_species` only.

## Part 1 — Motion crop

**Detector.** `MotionDetector.process()` already computes `changed` (thresholded diff × ROI mask). Additionally: dilate `changed` once (3×3) so a bird's parts join, `cv2.connectedComponentsWithStats`, drop components smaller than `MIN_BLOB_AREA = 4` px, and return the union bbox of the rest as `MotionResult.bbox: BBox | None` (lores coords). Computed on every frame (cheap at 320×240), only consumed on a trigger.

**Crop.** New pure `motion_crop(bbox, roi, frame_size) -> BBox | None` in `motion.py`:
- returns None (→ ROI fallback) when `bbox` is None or its area > 50 % of `roi_bbox(roi)`'s area (lighting change, swinging feeder);
- otherwise a **square** box centred on `bbox`, side = `max(2 × max(w, h), MIN_CROP_SIDE = 40)` lores px (≈ 410 main px), shrunk to the frame's short side if larger, then **shifted** (not shrunk) to lie inside the frame.

The square avoids today's aspect distortion when the crop is resized to 224×224. The ROI fallback keeps today's behaviour exactly (ROI bbox, 10 % pad, not square).

**Flow.** `capture.py` passes `result.bbox` through `_capture_still` → `store.add_capture(..., motion_bbox=…)` → `MotionEvent`. The pipeline (live, sweep and backfill) computes `motion_crop(capture.motion_bbox, cfg.motion.roi, lores_size)` from the stored box and the current ROI and passes it to `classify`; `CropLoader` gets a `pad` override so an already-padded motion crop is used with `pad = 0`. Rows without a box use the ROI crop as today.

**Debugging.**
- `birdy classify --capture <id>` classifies a stored capture with its stored crop (path and box from the DB) and prints which crop was used (`motion` / `roi`).
- `GET /captures/{id}/crop.jpg` serves the exact crop the model saw (JPEG, 404 unknown id); each grid tile links it as "model view".

## Part 2 — Visits and voting

**Grouping.** `assign_visit(capture)` (in a new `birdy/visits.py`, DB access through `Store`) runs whenever a photo's final `is_bird` becomes 1 and it has no visit. It finds visits with any photo within `visits.gap_seconds` (default 60) of this photo's `captured_at`:
- none → new visit;
- one → join it (extend `started_at`/`ended_at`);
- two or more (backfill or a correction filling a gap) → merge into the lowest id, move the photos, delete the others; the merged alert state is `sent` if any was sent, else `skipped` if any was skipped, else `pending` (never alert twice for one bird).

Photos the model calls non-bird get no visit and their final label is the model's (as today). Backfill never demotes, so a photo only stops being a bird through part 3's "not a bird", and it keeps its `visit_id` then.

**Vote.** `vote(photos, cfg) -> (species, confidence)` is pure:
- score(s) = Σ over the visit's photos of p(s) from each photo's `model_top5` (or `model_species`/`model_species_confidence` when `model_top5` is NULL);
- winner = highest score (ties: the species with the higher single-photo p, then alphabetical);
- reported only if some photo has p(winner) ≥ `species.threshold`; confidence = that max p. A one-photo visit therefore gives exactly today's label.

**Resolve.** `resolve_visit(visit_id) -> list[Change]` (`Change = (capture_id, old_species, new_species, captured_at)`), final label per photo, highest priority first:
1. the newest correction among the visit's photos (part 3);
2. the photo's own first choice when it differs from the winner and its p ≥ `visits.keep_own_threshold` (default 0.8) — two different birds at once;
3. the vote;
4. bird, no species.

`species_confidence` = the photo's own p for its final species from `model_top5` (0 if absent; NULL when no species), so publishing still ranks the photos the model is surest of. `confidence` stays `model_confidence`; `is_bird` = 1 unless the correction says not a bird. Writes `visits.species/species_confidence`. Called after every live classification, backfill update, correction, merge and visit close. The pipeline passes each `Change` to `Publisher.evaluate_day(species, local_day)` for both old and new species (new method; `evaluate(capture)` becomes a wrapper), so a photo moved away from a species frees its slot.

**Close and alert.** The pipeline worker checks every ~5 s (`_run` loop, not the 5-min sweep) for open visits with `ended_at` < now − `gap_seconds`: mark closed, resolve, then if `alert = 'pending'`:
- `ended_at` older than 10 min (restart, long backlog) → `skipped`, logged;
- visit has a species → `notifier.consider(best photo, visit prediction, captured_at, first_seen=store.count_visits(species) == 1)` → `sent` when queued, `skipped` when the policy declines;
- no species → `skipped`.

Best photo = highest p for the winner (ties: lower id). `consider` takes `first_seen` from the caller instead of counting photos: "new species" now means first **visit** with that species. Visits created from backfilled or historical photos start as `skipped`. The live path (`Pipeline.process`) no longer calls the notifier. Known limitation: a second bird kept by rule 2 does not get its own alert.

**Config** (`[visits]`, both live, on `/settings`): `gap_seconds = 60` (10–600), `keep_own_threshold = 0.8` (0.5–1.0).

**Grid.** Tiles stay one per photo; a tile in a multi-photo visit shows "visit · N photos". The grid's species filter and the badge show the final label.

## Part 3 — Corrections

**Storage.** On the marked photo: `manual_is_bird` (0/1), `manual_species` (common name or NULL), `manual_at` (UTC). `resolve_visit` applies the newest correction in the visit to every photo, so a merge of two corrected visits settles to the newest.

**Endpoint.** `POST /captures/{id}/label` with `label` = `not_bird` | `bird_unknown` | `species:<common name>` | `clear`, and `next` (same rule as the other tile buttons: `/` or `/?…`, else `/`). 404 unknown id, 400 for any other `label` or a species name not in the dropdown list. 303 to `next`.
- `not_bird`: every photo in the visit gets `is_bird = 0`, `species = NULL`; the photos keep `visit_id` so "back to model" restores them.
- `bird_unknown` / `species:…` on a photo without a visit: `assign_visit` creates (or joins) a visit with `alert = 'skipped'`.
- `clear`: NULL the photo's three manual columns, re-resolve.

Then `resolve_visit` and re-rank as in part 2. A correction never sends or re-sends an alert.

**Tile UI.** A **Fix** `<details>` form beside Delete/Publish: buttons Not a bird / Bird, species unknown / Back to model (only when corrected), and a species `<select>` + Set — the site's local list (part 4, most observations first), then a separator, then all model species not already listed. Reversible, so no two-tap confirm. A corrected tile shows a **you** badge and, muted, "model: <model_species> <p>". Visual rules per `DESIGN.md`.

**Retention.** `purge_non_birds` skips photos with `manual_at` set (they are the future training examples); other photos in that visit follow the normal rule.

## Part 4 — Per-site species list

**Fetch.** `birdy/species_list.py`. `GET https://api.inaturalist.org/v1/observations/species_counts?lat=…&lng=…&radius=<km>&taxon_id=3&quality_grade=research&locale=en&per_page=500` (paged until exhausted), via `urllib` with a 20 s timeout. Keep species with ≥ 3 observations (`latin = taxon.name`, `common = taxon.preferred_common_name or latin`, `count`). If fewer than 20 remain, retry once at 4× the radius. Saved atomically to `data/species/local.json`: `{fetched_at, lat, lon, radius_km, species: [{latin, common, count}]}`.

**When.** A one-shot background thread at service start fetches when the file is missing, older than 90 days, or its lat/lon/radius differ from the config. Failure: one warning, keep the existing file (or none). `birdy species-list [--refresh]` prints the list, model coverage and age; `--refresh` fetches now. No `site.lat/lon` → no list (all 964 species, as today), and `/settings` shows "set a location to enable the local species list".

**Config** (`[species]`): `region_radius_km = 50` (restart-only; changing it triggers a refetch on start), `include = []`, `exclude = []` (scientific names; live; edited on `/settings` like the Telegram watchlist).

**Mask.** Allowed = model labels whose `latin` (case-insensitive) is on the list, plus `include`, minus `exclude`. `InatBirdClassifier.top` sets disallowed classes to 0 and takes the top 5 of what remains. **No renormalisation**: the uint8 output is truncated (sums to ~0.8–0.95), and rescaling a small allowed mass would fabricate confidence that passes `species.threshold`. `background` is unaffected; the gate is unaffected. With no list, every class is allowed.

**Coverage.** `/species` shows a short note listing local species the model cannot recognise (on the list, no matching label — e.g. Sardinian Warbler, Iberian Magpie). They remain choosable in the correction dropdown.

**History.** The allowed set has a hash; when it changes (first fetch, refresh, include/exclude edit) `species_checked = 0` is set for photos with `model_is_bird = 1`, and the existing backfill reclassifies them (50 per sweep, yielding to live events) and re-resolves their visits. Backfilled visits keep `alert = 'skipped'`.

## Testing (all on the Mac)

- Motion: a small square moving inside the ROI gives its box; specks below `MIN_BLOB_AREA` are ignored; a whole-ROI brightening gives a box > 50 % → `motion_crop` None; `motion_crop` square, minimum side, frame-edge shifting; pipeline passes the stored box; `/captures/{id}/crop.jpg`; `birdy classify --capture`.
- Visits: grouping, joining, merging (photos moved, alert state), vote with top 5 and NULL `model_top5`, tie-breaks, rule-2 keep-own, resolve priority, `Change` list → `evaluate_day` for old and new species; closing with a fake clock (on time → alert, > 10 min late → skipped, backfill → skipped, no species → skipped); first-visit "new species".
- Corrections: each `label`, 400/404, `next` handling, not-bird and back, non-bird photo → new visit, newest correction wins after a merge, no alert sent, retention skips marked photos.
- Species list: parse a saved iNaturalist response (fixture), ≥ 3 filter, radius retry, refetch conditions, failure keeps the old file, mask without renormalisation, include/exclude, hash change queues the backlog.
- Migration on an old-format DB: columns, `model_*` copied, visits built and resolved, all alerts `skipped`, idempotent on second open.
- `test_config_to_dict_covers_every_field` covers the new keys.

## Docs to update

`birdy.toml.example` (new keys), `CLAUDE.md` (visits, final vs model columns, corrections, species list, new endpoints and CLI), `PRODUCT.md` if it describes labels.
