# Recognition improvements — design

Date: 2026-10-05. Builds on the capture core, species recognition (2), Telegram alerts (6) and publishing (3) specs.

## Why

The species model sees the box around the whole ROI polygon (`Pipeline.process` → `roi_bbox(cfg.motion.roi)`). At the feeder's 1.5 m a tit is ~210 px wide in the 3280 px still, but after the ROI crop is squeezed to 224×224 it is ~25–40 px — too small to identify. Each capture is also judged alone (labels flicker within one visit), there is no way to correct a wrong label, and the 965-class model guesses species from other continents.

Four parts:

1. **Motion crop** — classify the area that moved, not the whole ROI.
2. **Visits and voting** — group a bird's captures into a visit, decide one label per visit, alert once per visit.
3. **Corrections** — the user's labels, stored next to the model's, applied per visit.
4. **Per-site species list** — each Pi limits the model to species recorded near its own location.

Out of scope: learning from corrections (embeddings + kNN), replacing either model, audio. The data they need is kept (raw model output, corrected photos never auto-deleted).

## Build order — three plans

- **Plan A — Part 1.** Independent of the rest; ship and measure first.
- **Plan B — Parts 2 + 3.** One data-model change (`model_*` columns, visits, resolve, per-visit alerts, migration, corrections). May split into B1 (schema, visits, alerts, migration) and B2 (correction endpoint, tile UI, retention).
- **Plan C — Part 4.** Needs B (the mask is applied when voting).

Each plan ends by updating `CLAUDE.md` and `birdy.toml.example`.

## Data model

`captures` gains (all nullable; added by migration):

| Column | Plan | Meaning |
|---|---|---|
| `motion_x`, `motion_y`, `motion_w`, `motion_h` | A | Raw motion box at the trigger frame, **normalised 0..1** of the lores frame (so a later `lores_size` change can't misread it). NULL = none |
| `model_is_bird`, `model_confidence`, `model_species`, `model_species_confidence` | B | What the classifier decided for this photo at classification time (`model_confidence` = gate P(bird)) |
| `model_top10` | B | JSON `[[latin, p], …]`: the species model's **unmasked** top 10 (`background` excluded). Lets the mask and thresholds be re-applied without inference |
| `visit_id` | B | FK → `visits.id`; NULL for photos that are not birds and were never corrected to bird |
| `manual_is_bird`, `manual_species`, `manual_at` | B | The user's correction, copied onto every photo of the visit at correction time; NULL = none |

`is_bird`, `confidence`, `species`, `species_confidence` keep their names and every existing reader (grid filters, species dropdown, `/species`, `last_species`, `rank_day`, manifest, retention) keeps reading them, but they now hold the **final** label. Writers:

- `update_classification` writes **only** `model_*`, `model_top10`, `species_checked`, `status`. It never touches the final columns.
- Final columns are written only by `resolve_visit` (photos in a visit) and `resolve_single` (photos without a visit: final = model, i.e. today's behaviour).
- The backfill's "never demote" rule (`c.is_bird or pred.is_bird`, pipeline.py:142) is removed: a photo in a visit is a bird unless corrected, which is the same guarantee.

New table:

```sql
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,      -- UTC ISO, first photo
  ended_at TEXT NOT NULL,        -- UTC ISO, last photo
  closed INTEGER NOT NULL DEFAULT 0,
  species TEXT,                  -- resolved visit label (common name; NULL = bird, no species)
  species_confidence REAL,
  alert TEXT NOT NULL DEFAULT 'pending'   -- pending | sent | skipped
);
CREATE INDEX IF NOT EXISTS visits_time ON visits(ended_at);
CREATE INDEX IF NOT EXISTS visits_open ON visits(closed, ended_at);
CREATE INDEX IF NOT EXISTS captures_visit ON captures(visit_id);
```

**Species names.** The final `species` column stays the common name (what the UI, gallery and alerts show). Votes, the mask and `model_top10` work on latin names; `birdy/species.py` maps latin → common from the model labels, and the species list (Part 4) supplies names for species the model doesn't know.

## Part 1 — Motion crop (Plan A)

**Detector.** `MotionDetector.process()` already computes `changed` (thresholded diff × ROI mask). Additionally: dilate `changed` once (3×3) so a bird's parts join, `cv2.connectedComponentsWithStats`, drop components smaller than `MIN_BLOB_AREA = 4` px, and return the union bbox of the rest as `MotionResult.bbox: BBox | None` (lores px). Computed on every frame (cheap at 320×240), only consumed on a trigger.

**Crop.** New pure `motion_crop(bbox, roi, frame_size) -> BBox | None` in `motion.py`:
- None (→ ROI fallback) when `bbox` is None or its area > 50 % of `roi_bbox(roi)`'s area (lighting change, swinging feeder). The test uses the *current* ROI, so a backfill after an ROI edit may decide differently from capture time — acceptable;
- otherwise a **square** box centred on `bbox`, side = `max(2 × max(w, h), MIN_CROP_SIDE = 40)` lores px (≈ 410 main px), capped at the frame's short side, then **shifted** (not shrunk) to lie inside the frame.

The still is taken a few hundred ms after the trigger frame (capture.py:83); the 2× square absorbs that movement. The square avoids today's aspect distortion when the crop is resized to 224×224; `config.py` validates that `main_size` and `lores_size` have the same aspect ratio (± 1 %) so a lores square is a main square. The ROI fallback keeps today's behaviour exactly (ROI bbox, 10 % pad, not square).

**Flow.** `capture.py` passes `result.bbox` (normalised) through `_capture_still` → `store.add_capture(..., motion_box=…)` → `MotionEvent`. The pipeline (live, sweep, backfill) de-normalises with the process's pinned `lores_size`, computes `motion_crop` with the current ROI, and passes it to `classify`; `CropLoader` gets a `pad` override so an already-padded motion crop is used with `pad = 0`. Rows without a box use the ROI crop as today.

**Debugging.**
- `birdy classify --capture <id>` (positional `image` becomes `nargs="?"`, cli.py:35) classifies a stored capture with its stored crop and prints which crop was used (`motion` / `roi`).
- `GET /captures/{id}/crop.jpg` serves the exact crop the model saw (JPEG, 404 unknown id, decoded in `run_in_threadpool`); each grid tile links it as "model view".

## Part 2 — Visits and voting (Plan B)

All visit logic lives in `birdy/visits.py` (`Visits`, one instance shared by the pipeline and the web, built in `cmd_run`). One `threading.RLock` in `Visits` serialises assign/merge/resolve/close/correct; each operation runs in one SQLite transaction through `Store`. `Store._lock` stays per statement underneath.

**Labelling one photo.** Pure `label_photo(model, mask, cfg) -> (is_bird, confidence, species_latin, p)` applies today's `TwoStageClassifier` rules (species threshold, rescue) to the stored gate P(bird) and the masked `model_top10`. Used at classification time to set `model_*` and decide visit membership. Bird/non-bird membership is decided once, at classification time; later threshold or mask changes re-resolve species but don't regroup photos (as today, where gate-threshold changes only affect new photos).

**Grouping.** `assign_visit(capture)` runs when a photo becomes a bird (classification or correction) and has no visit. Candidates: visits whose `[started_at − gap, ended_at + gap]` contains the photo's `captured_at` (indexed), **excluding visits whose photos are all corrected to not-a-bird**:
- none → new visit (`alert = 'pending'` when live, `'skipped'` from backfill, migration or a correction);
- one → join it (extend `started_at`/`ended_at`);
- two or more → merge into the lowest id, move the photos, delete the others; merged alert state = `sent` if any was sent, else `skipped` if any was, else `pending` (never alert twice for one bird).

A joining photo takes the visit's existing correction (copied, see Part 3).

**Vote.** Pure `vote(photos, mask, cfg) -> (species_latin, confidence) | None`:
- score(s) = Σ over the visit's photos of p(s) from each photo's masked `model_top10` (old rows without it: `model_species`/`model_species_confidence`);
- winner = highest score (ties: higher single-photo p, then alphabetical);
- reported only if some photo has p(winner) ≥ `species.threshold`; confidence = that max p.

Intended: voting makes a visit **consistent**, it does not raise confidence. Five photos at 0.45 give no species, as each alone would today; a one-photo visit gives exactly today's label.

**Resolve.** `resolve_visit(visit_id) -> list[Change]` (`Change = (capture_id, old_species, new_species, captured_at)`). Final label per photo, highest priority first:
1. the photo's own correction (Part 3);
2. the photo's own masked first choice when it differs from the winner and its p ≥ `KEEP_OWN_THRESHOLD = 0.8` (two different birds at once; a constant);
3. the vote;
4. bird, no species.

`is_bird` = 1 unless corrected to not-a-bird; `confidence` = `model_confidence`; `species_confidence` = the photo's own p for its final species, or for a corrected species the photo's gate P(bird) (the model may not know the species at all; P(bird) ranks the clearest shots and lets corrected visits auto-publish). Writes `visits.species/species_confidence`.

Publishing: for every `Change`, `resolve_visit` first sets `published = NULL` on rows that were `auto` (so a photo leaving a species, or becoming not-a-bird, can't stay in the gallery; `forced`/`hidden` are kept), then the caller passes each `Change` to `Publisher.evaluate_day(species, local_day)` for the old and new species (new method; `evaluate(capture)` becomes a wrapper).

Called after every live classification, backfill update, correction, merge, delete and visit close.

**Close and alert.** The pipeline worker checks every ~5 s (in `_run`, not the 5-min sweep):
- **close** open visits with `ended_at` < now − `visits.gap_seconds`; resolve;
- **alert** a visit with `alert = 'pending'` when it closes **or** when it has been open `ALERT_AFTER_SECONDS = 300` (a pigeon sitting for 20 min alerts after 5, with the vote so far). Then:
  - `ended_at` older than 10 min at decision time (restart, backlog) → `skipped`, logged;
  - no species → `skipped`;
  - else `notifier.consider(best photo, visit prediction, captured_at, first_seen=store.count_visits(species) == 1)` → `sent` when queued, `skipped` when the policy declines.

Best photo = highest p for the winner (ties: lower id). `consider` takes `first_seen` from the caller: "new species" means first **visit** with that species. `Pipeline.process` no longer calls the notifier. Known limitation: a second bird kept by rule 2 doesn't get its own alert.

**Deletes.** `delete_capture`, `delete_non_birds` and `purge_non_birds` call `Visits.photo_removed(visit_id)`: re-resolve the visit, or delete it when it has no photos left (so `count_visits` and time-joins don't see it).

**Config** (`[visits]`, live, on `/settings`): `gap_seconds = 60` (10–600).

**Grid.** Tiles stay one per photo; a tile in a multi-photo visit shows "visit · N photos". The species filter and the badge show the final label.

**Migration** (Plan B; on `Store` open, guarded by `PRAGMA user_version`):
- back up first with `sqlite3.Connection.backup` to `data/birdy.db.bak`;
- `BEGIN IMMEDIATE`, re-check `user_version` inside (the service, `birdy purge` and `birdy publish` all open the Store), add columns, copy `is_bird/confidence/species/species_confidence` into `model_*`, build visits over all history in `captured_at` order (all `closed = 1`, `alert = 'skipped'`), resolve every visit, bump `user_version`, commit;
- set `store.migrated_visits = True`; `cmd_run` then calls `publisher.evaluate_all()` when publishing is enabled.

Old rows have `model_top10` NULL and vote with `model_species` only. Idempotent: a second open does nothing.

## Part 3 — Corrections (Plan B)

**Storage.** `manual_is_bird` (0/1), `manual_species` (latin, or NULL for "bird, species unknown"), `manual_at` (UTC). A correction is **copied onto every photo currently in the visit**; resolve rule 1 reads only the photo's own columns. So a later bird joining the visit by time takes the visit's correction only if the visit still has birds (visits that are entirely not-a-bird are excluded from joins), and a correction can't leak into a different, later bird.

**Endpoint.** `POST /captures/{id}/label` with `label` = `not_bird` | `bird_unknown` | `species:<latin>` | `clear`, and `next` (same rule as the other tile buttons: `/` or `/?…`, else `/`). 404 unknown id, 400 for any other `label` or a latin name not in the dropdown list. 303 to `next`. Runs under the `Visits` lock.
- `not_bird`: every photo in the visit gets the correction; they keep `visit_id` so "back to model" restores them.
- `bird_unknown` / `species:…` on a photo without a visit: `assign_visit` (alert `skipped`), then copy the correction.
- `clear`: NULL the manual columns on every photo of the visit, re-resolve.

Then resolve + re-rank as in Part 2. A correction never sends or re-sends an alert.

**Tile UI.** A **Fix** `<details>` form beside Delete/Publish: buttons Not a bird / Bird, species unknown / Back to model (only when corrected), and a species `<select>` + Set — the site's local list (Part 4, most observations first), a separator, then all model species not already listed (before Part 4 ships: model species only). Reversible, so no two-tap confirm. A corrected tile shows a **you** badge and, muted, "model: <model species> <p>". Visual rules per `DESIGN.md`.

**Retention.** `purge_non_birds` skips photos with `manual_at` set (future training examples).

## Part 4 — Per-site species list (Plan C)

**Fetch.** `birdy/species_list.py`. `GET https://api.inaturalist.org/v1/observations/species_counts?lat=…&lng=…&radius=<km>&taxon_id=3&quality_grade=research&locale=en&per_page=500`, paged until exhausted, ≤ 1 request/s, `User-Agent: birdy (+https://birdy.akibeo.com)`, 20 s timeout (all verified against the live API on 2026-10-05: Sintra at 50 km returns 415 species). Keep `taxon.rank == "species"` (results include hybrids) with ≥ 3 observations: `latin = taxon.name`, `common = taxon.preferred_common_name or latin`, `count`. Fewer than 20 → retry once at 4× radius. Saved atomically to `data/species/local.json`: `{fetched_at, lat, lon, radius_km, species: [{latin, common, count}]}`.

**When.** A one-shot background thread at service start fetches when the file is missing, older than 90 days, or its lat/lon/radius differ from the config. Failure: one warning, keep the existing file (or none). `birdy species-list [--refresh]` prints the list, model coverage and age; `--refresh` fetches now. No `site.lat/lon` → no list (all species, as today), and `/settings` shows "set a location to enable the local species list".

**Matching to model labels.** The 2017 labels use older taxonomy (Sintra check: 355 local species, only 200 match by latin; e.g. *Linaria cannabina* is labelled `Carduelis cannabina`, *Curruca communis* is `Sylvia communis`; in the US, Ruby-crowned Kinglet moved from *Regulus* to *Corthylio*). A label is on the list when **either**:
- its latin, cut to the first two words (labels include subspecies like `Columba livia domestica`), equals a list latin (casefold); or
- its common name equals a list common name, normalised (casefold, hyphens → spaces).

The result is a set of allowed **label** latins, plus `include` and minus `exclude` (`species.include`/`exclude`: latin names, live, edited on `/settings` like the Telegram watchlist; matched the same way).

**Mask.** Applied in `label_photo` and `vote` to `model_top10`: drop disallowed species. **No renormalisation**: the uint8 output is truncated (sums to ~0.8–0.95), and rescaling a small allowed mass would fabricate confidence that passes `species.threshold`. `background` and the gate are unaffected. No list → everything allowed.

**Applying a change.** Because `model_top10` is stored unmasked, a new list or an include/exclude edit only re-resolves: every visit is resolved again (SQL + Python, seconds), species change, publishing re-ranks the changed days. No re-inference, no `species_checked` reset. Bird/non-bird membership is not revisited (Part 2). Rows classified before Plan B have no `model_top10`; they vote with their single stored `model_species`, masked (a species off the list counts as no species).

**Coverage.** `/species` shows a short note listing local species the model can't recognise (on the list, no matching label — e.g. Sardinian Warbler, Iberian Magpie). They remain choosable in the correction dropdown; their common name comes from the list.

**Config** (`[species]`): `region_radius_km = 50` (restart-only; a change triggers a refetch on start), `include = []`, `exclude = []` (live).

## Testing (all on the Mac)

- **A:** a small square moving inside the ROI gives its box; specks below `MIN_BLOB_AREA` ignored; whole-ROI brightening → `motion_crop` None; square, minimum side, frame-edge shifting; normalised box round-trip; aspect-ratio validation; pipeline passes the stored box; `/captures/{id}/crop.jpg`; `birdy classify --capture`.
- **B:** `update_classification` never writes final columns; `label_photo` matches today's rules; grouping, joining, merging (photos moved, alert state); not-a-bird visits excluded from joins (a bird 30 s after a corrected visit gets a new visit); vote with top 10 and NULL `model_top10`, ties, keep-own; resolve priority; `auto` cleared when species changes; `Change` → `evaluate_day` for old and new species; close/alert with a fake clock (on close, at 300 s open, > 10 min late, backfill, no species); first-visit "new species"; deletes re-resolve / drop empty visits; concurrent correction + pipeline resolve (threads) end consistent; corrections (each `label`, 400/404, `next`, non-bird → visit, clear, no alert, joining photo inherits the correction); retention skips corrected photos; migration on an old-format DB (backup, columns, `model_*`, visits, `skipped`, idempotent, `migrated_visits`).
- **C:** parse a saved iNaturalist response (fixture incl. a hybrid), ≥ 3 filter, radius retry, refetch conditions, failure keeps the old file; label matching by two-word latin and by common name (Linnet, Common Whitethroat, a subspecies label); mask without renormalisation; include/exclude; a list change re-resolves without inference.
- `test_config_to_dict_covers_every_field` covers the new keys.
