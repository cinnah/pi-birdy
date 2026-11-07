# Visits and corrections — design

Date: 2026-11-07. This is Plan B (Parts 2 + 3) of `2026-10-05-recognition-improvements-design.md`, rewritten for the BioCLIP 2 species model (`2026-11-06-bioclip-species-design.md`) and the detector gate (`2026-11-06-detector-gate-design.md`), both deployed. Where this document and the October one differ, this one wins.

## Why

Each photo is judged alone, so a visit of ten photos can carry three different names, and the only correction a person can make is "not a bird". With BioCLIP 2 the name is right on most photos of a visit (Black Redstart on 99 of 117), but the odd photo still flips to a lookalike, and when the model is wrong there is no way to say so. Grouping photos into **visits** gives one label per bird visit, one alert per visit, and one click to correct all of a visit's photos.

## Scope

In: visits (grouping, merging, closing), a per-visit species vote in embedding space, per-visit alerts, corrections (`not_bird`, `bird_unknown`, a species name, `clear`) with a Fix panel on the grid, publish-slot moves on every label change, deletes keeping visits consistent, a one-time migration of history, `[visits] gap_seconds` on the settings page. Out: changing how often BioCLIP runs (still once per bird photo), learning from corrections, the species page beyond what the final columns already give it, multi-bird visits beyond the keep-own rule.

## Data model

`captures` gains (nullable, added by the existing `ALTER` loop):

| Column | Meaning |
|---|---|
| `visit_id INTEGER` | FK → `visits.id`; NULL for non-birds never corrected to bird |
| `embedding BLOB` | the photo's L2-normalised BioCLIP image embedding, 768 × float16 (1.5 KB); NULL until the species model has seen the photo |

The existing `model_is_bird`, `model_confidence`, `model_species`, `model_species_confidence`, `manual_is_bird`, `manual_species`, `manual_at` columns keep their names. `model_*` now always holds the classifier's verdict for the photo (not only after a correction); `manual_*` holds a person's correction, copied onto every photo of the visit at correction time; `manual_species` is the bank's **common name** (what the UI shows), with the iNat id resolved at correction time.

New table:

```sql
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,      -- UTC ISO, first photo
  ended_at TEXT NOT NULL,        -- UTC ISO, last photo
  closed INTEGER NOT NULL DEFAULT 0,
  species TEXT,                  -- resolved label (common name); NULL = bird, no species
  species_confidence REAL,       -- pooled probability of that label
  alert TEXT NOT NULL DEFAULT 'pending'   -- pending | sent | skipped
);
CREATE INDEX IF NOT EXISTS visits_time ON visits(ended_at);
CREATE INDEX IF NOT EXISTS visits_open ON visits(closed, ended_at);
CREATE INDEX IF NOT EXISTS captures_visit ON captures(visit_id);
```

**Writers** (as implemented, `update_classification` differs: see Implementation notes). `update_classification` writes **only** `model_*`, `embedding`, `det_*`, `species_checked`, `detector_checked`, `status`; it never touches `is_bird/confidence/species/species_confidence` again. Those final columns are written only by `Visits.resolve(visit_id)` (photos in a visit) and `Visits.resolve_single(capture_id)` (a photo without a visit: final = model). Every existing reader (grid filters, species dropdown, `/species`, `last_species`, `rank_day`, manifest, retention, alerts) keeps reading the final columns.

## Classifier and pipeline

`Prediction` gains `embedding: np.ndarray | None` (set when the species model ran). `BioClipSpecies.top` already computes the embedding; `classify_image` keeps it. `Pipeline.process` and `_backfill` store it with the `model_*` verdict, then hand the row to `Visits`:

- live: `visits.photo_classified(capture, live=True)`;
- backfill: `visits.photo_classified(capture, live=False)`.

`Pipeline.process` **no longer calls the notifier**; alerts come from visit closing (below). The backfill's never-demote rule and the known-bird rule stay: `model_is_bird` is the fresh verdict, but a stored bird stays a bird (`Visits` treats `manual_is_bird`/`is_bird` as "bird" membership — see Grouping).

## Visits — `birdy/visits.py`

One `Visits` object (built in `cmd_run`, shared by the pipeline and the web), with one `threading.RLock` around assign/merge/resolve/close/correct; each operation is one SQLite transaction through `Store` (the store lock stays per statement underneath). Pure functions take data and return decisions; the class does the I/O.

**Species scoring.** `Visits` holds the `BioClipSpecies` bank and the `SpeciesList` (same objects as the classifier). `score(embedding) -> list[(id, p)]` is the softmax over the current candidate ids of `logit_scale · E_candidates · e` — the same arithmetic as `BioClipSpecies.top`, factored into a pure `birdy/species.py::score_candidates(bank, cand_rows, emb)` used by both. No inference anywhere in `Visits`.

**Membership.** A photo is a *bird member* when `manual_is_bird = 1`, or when `manual_is_bird IS NULL` and `model_is_bird = 1`, or when it is a stored bird the backfill never demotes (`is_bird = 1` with `manual_at IS NULL`). A photo corrected to `not_bird` keeps its `visit_id` (so **Back to model** can restore it) but is not a bird member.

**Grouping** — `assign(capture)` runs when a photo becomes a bird member and has no visit. Candidates: visits whose `[started_at − gap, ended_at + gap]` contains `captured_at`, **excluding visits with no bird members** (all corrected to not-a-bird). None → new visit (`alert = 'pending'` when live, `'skipped'` from backfill, migration or a correction); one → join (extend the span); two or more → merge into the lowest id (move photos, delete the others; merged `alert` = `sent` if any was sent, else `skipped` if any was, else `pending`). A joining photo takes the visit's existing correction (copied) if the visit has one. `gap = cfg.visits.gap_seconds` (live).

**Vote** — pure `vote(members, cand) -> (species_id, p, pooled) | None`: `pooled = normalise(mean(embeddings))` over bird members that have an embedding; `score(pooled)`; the winner is reported only when `p ≥ species.threshold`. No embeddings yet (fresh migration) → `None`. Ties: higher own-photo p, then alphabetical. Intended: the vote makes a visit **consistent** and a little more confident than single blurry frames, but it cannot name a visit that no frame supports.

**Resolve** — `resolve(visit_id) -> list[Change]` (`Change = (capture_id, old_species, new_species, captured_at)`). Final label per photo, highest priority first:

1. the photo's own correction: `manual_is_bird = 0` → non-bird, no species; `manual_is_bird = 1` → bird with `manual_species` (or none);
2. the photo's own top-1 when it differs from the vote and its own p ≥ `KEEP_OWN = 0.8` (a second bird in frame); a constant;
3. the vote;
4. bird, no species.

`is_bird` = 1 for bird members, 0 for not-bird corrections; `confidence` = `model_confidence`; `species_confidence` = the photo's own p for its final species (its own scoring), or for a corrected species the photo's `model_confidence` (lets corrected visits auto-publish and ranks the sharpest shot). Writes `visits.species/species_confidence`.

Publishing, for every `Change` with `old ≠ new`: rows that were `auto` get `published = NULL` first (`forced`/`hidden` kept), then `Publisher.evaluate_day(old_species, captured_at)` and `evaluate_day(new_species, captured_at)` (new method; `evaluate(capture)` becomes a wrapper around it). Called after every live classification, backfill update, correction, merge, delete and close.

**`resolve_single(capture_id)`** — a classified photo with no visit (a non-bird): final = model verdict (today's behaviour), `species = model_species` only if it is a bird — i.e. non-birds get no species in the final columns (the grid's "model: X" muted hint reads `model_species` directly).

**Close and alert.** The pipeline worker calls `visits.tick(now)` every ~5 s (in `_run`, not the 5-min sweep):

- close open visits with `ended_at < now − gap`; resolve each;
- alert a visit with `alert = 'pending'` when it closes **or** has been open ≥ `ALERT_AFTER = 300 s` (a bird that sits for twenty minutes alerts after five, with the vote so far). Then: `ended_at` older than 10 min at decision time (restart, backlog) → `skipped`, logged; no species → `skipped`; else `notifier.consider(best photo, species, p, captured_at, first_seen = store.count_visits_with_species(species) == 1)` → `sent` when queued, `skipped` when the policy declines. Best photo = highest own p for the visit's species (ties: lower id). `TelegramNotifier.consider` gains `first_seen` as a parameter and drops its own count; the alert caption is unchanged. Known limitation: a second bird kept by rule 2 does not get its own alert.

**Deletes.** `delete_capture`, `delete_non_birds` and `purge_non_birds` tell `Visits` (`photo_removed(visit_id)`): re-resolve the visit, or delete it when it has no photos left (so `count_visits_with_species` and time-joins don't see it). The web routes already call `publisher.forget`; keep that.

## Corrections — Part 3

**Endpoint.** `POST /captures/{id}/label` with `label` = `not_bird` | `bird_unknown` | `species:<name>` | `clear` and `next` (same rule as today). 404 unknown id; 409 unless `status = 'done'`; 400 for any other `label` or a `<name>` the bank cannot resolve (`Bank.resolve`, latin or common, case-insensitive; the stored `manual_species` is the bank's common name). Runs under the `Visits` lock:

- `not_bird`: every photo in the visit (or just this photo when it has no visit) gets `manual_is_bird = 0, manual_species = NULL, manual_at = now`; they keep `visit_id`.
- `bird_unknown` / `species:…` on a photo without a visit (a non-bird the model missed, or a lone photo): `assign` first (alert `skipped`), then copy the correction (`manual_is_bird = 1`, `manual_species`) onto every photo of the visit.
- `clear`: NULL the `manual_*` columns on every photo of the visit; re-resolve.

Then resolve + re-rank as above. A correction never sends or re-sends an alert. The log line names the photo, the visit size and the label.

**Grid.** Each bird tile (and each corrected tile) gets a **Fix** `<details>` beside Delete/Publish: buttons **Not a bird** / **Bird, species unknown** / **Back to model** (only when corrected) and a species `<input list="species-names">` + **Set**. The `<datalist>` holds the local list (most observations first) and the species seen so far; the server accepts any bank name, so a typed name outside the list works too. A corrected tile shows a **you** badge and, muted, `model: <model_species> <p>`. Tiles in a multi-photo visit show `visit · N` with a link to `/?visit=<id>` (a new grid filter showing just that visit, newest first; `FILTERS` gains nothing — it is a `visit` query parameter handled like `species`). No two-tap confirm: corrections are reversible. Visual rules per `DESIGN.md`.

**Retention.** `purge_non_birds` already skips `manual_at IS NOT NULL`; unchanged.

## Config

```toml
[visits]
gap_seconds = 60           # photos of one bird within this gap form a visit (10–600); live
```

`config.py` dataclass + validation, `birdy.toml.example`, `settings.py` (`SettingsPatch.visits_gap_seconds`, `config_to_dict`, `apply_patch`, `FIELD_KEYS`, `parse_settings_form`), a `num()` row on `/settings` under a new "Visits" section. `test_config_to_dict_covers_every_field` covers it.

## Migration (one-time, on `Store` open, guarded by `meta.visits_version = '1'`)

1. `sqlite3.Connection.backup` to `data/birdy.db.bak-visits` first.
2. `BEGIN IMMEDIATE`; re-check `meta.visits_version` inside (the service, `birdy purge` and `birdy publish` all open the Store); add the columns and the table; copy `is_bird/confidence/species/species_confidence` into `model_*` **only where `manual_at IS NULL`** (corrected rows already hold the model's verdict in `model_*`); build visits over all history in `captured_at` order with the configured gap (all `closed = 1`, `alert = 'skipped'`); set `meta.visits_version`; commit.
3. Rows have no `embedding` yet, so `species_checked = 0` is set on every uncorrected bird (the same mechanism as a model change); the backfill fills embeddings (~119 photos, ~7 min) and `Visits.photo_classified(live=False)` re-resolves each visit as its photos come back. Until then a visit resolves by the stored `model_species` of its members (majority, ties by confidence) — so nothing goes unnamed during the pass.
4. `store.migrated_visits = True`; `cmd_run` then calls `publisher.evaluate_all()` when publishing is enabled.

Idempotent: a second open does nothing.

## Pi operations

Memory +0 (embeddings are stored, not held); disk +1.5 KB per bird photo. The tick runs every 5 s on the pipeline thread: one indexed query, cheap. The grid gets one extra `COUNT` per visit per page (cached per render). Alerts: after the migration nothing alerts for history; the first live visit of a species alerts as "new" only if no visit of it exists — so the Black Redstart is not "new".

## Testing (all on the Mac)

- `score_candidates` matches `BioClipSpecies.top` on the stand-in model; `vote` with hand-built unit embeddings (pooled winner; below-threshold → None; ties); `KEEP_OWN`.
- Grouping: new / join / merge (photos moved, alert state merged as specified); not-a-bird visits excluded from joins; a joining photo inherits the visit's correction; gap from `live`.
- Resolve priority (each of the four rules); `auto` cleared and `evaluate_day` called for old and new species on a change; `resolve_single` for non-birds.
- Tick with a fake clock: close after gap; alert at close; alert at 300 s open; > 10 min late → skipped; no species → skipped; backfill/migration visits never alert; `first_seen` = first visit; `consider` receives the best photo.
- Corrections: each `label`; 400 for an unknown name and other labels; 404/409; `next` rule; a non-bird corrected to a species gets a visit with `alert = 'skipped'`; `clear` re-resolves; a correction never alerts; the whole visit is updated; the datalist contains the local list.
- Deletes re-resolve or drop empty visits; `count_visits_with_species` ignores deleted visits.
- Pipeline: `process` no longer alerts; `update_classification` never writes final columns; backfill feeds `Visits`.
- Migration on a copy of an old-format DB: backup file exists; columns/table; `model_*` copied except for corrected rows; visits built with the right spans; `alert = 'skipped'`; `species_checked` reset for birds without embeddings; idempotent; `migrated_visits`.
- Settings: `visits.gap_seconds` round trip, range validation, row on the page.
- Web: the Fix panel markup per tile state; `?visit=` filter; grid shows `visit · N`.

## Implementation notes / deviations

Implemented on branch `visits`, 2026-11-07. Where the code differs from the text above:

- **Writers.** `Store.update_classification` still writes both `model_*` and the final columns (what the existing tests and a lone photo need). The pipeline uses the new `Store.store_verdict` (`model_*`, `embedding`, `det_*`, the checked flags, `status`; a None embedding keeps the stored one); the final columns are written only by `Visits` through `Store.write_final`.
- **Membership** has two clauses, not three: `manual_is_bird = 1 OR (manual_is_bird IS NULL AND model_is_bird = 1)` (`store._BIRD_MEMBER`). The backfill's never-demote rule is folded into the verdict instead: it stores `model_is_bird = old bird OR new bird`.
- **Vote** returns `Vote(species, p)`; ties are not broken explicitly. The majority fallback runs only when there is no bank or no member has an embedding; when some members have one, the vote uses those alone.
- **Default `Visits`.** `Pipeline(visits=None)` builds a `Visits` with no bank and no species list (model-species majority, no embeddings needed); `cmd_run` passes the real one (bank via `getattr`, so test doubles get `bank=None`).
- **Tick** every 5 s via `Pipeline(tick_interval=)` (`TICK_INTERVAL`), called from `_run` and also between backfill rows and unlabelled rows (`_maybe_tick`), so a long sweep doesn't hold back closes and alerts.
- **Unlabelled sweep** (not in the text): each 5-min sweep also resolves `done` rows with no final label (`store.unresolved_done`, a crash between `store_verdict` and `Visits`) with `live=True`; the 10-min late rule keeps old ones from alerting.
- **Publishing order and dedup.** Final columns are written first, publish states second, so a re-rank in between can't hand `auto` back to a photo about to lose it. `Visits._publish` re-ranks each (species, local day) once per resolve (a 30-photo rename is two `evaluate_day` calls), and also re-ranks a day when a photo keeps its name but its confidence changes. `evaluate_day` itself does not dedupe.
- **Non-bird leaves the gallery in any state.** A photo whose final label becomes non-bird has `published` cleared whatever it was (`forced` and `hidden` too, unlike a rename, which keeps them), and the mirror is told to sync.
- **`first_seen`.** `count_visits_with_species` counts only visits that still have a bird member, so a visit emptied by deletes or by `birdy purge` (which runs without `Visits` and leaves such a visit unrevisited) doesn't count.
- **Corrections.** The Fix panel is on every `done` tile, not only bird and corrected ones (**Not a bird** shows on birds only). The form posts `label=species` with the name in `species_name`; `species:<name>` is also accepted. Without a `Visits` instance, `species`/`bird_unknown` are 503 and `not_bird`/`clear` fall back to the per-photo `store.mark_not_bird`/`clear_correction`. On join or merge the most recent correction (by `manual_at`, then id) goes to every photo of the visit. `clear` also detaches photos that are non-members once the correction is gone and resolves each alone.
- **Migration** runs from `cmd_run` (`Store.migrate_visits(gap_seconds)`, returns the number of visits built), not on `Store` open, so `birdy purge` and `birdy publish` never migrate. The columns and the table come from `Store`'s schema and `ALTER` loop at open, not from the migration transaction. Migrated visits take their species from the members' most common final name. A row corrected to not-a-bird gets no visit (it is not a bird member). `cmd_run` logs the number of photos queued and an ETA.
- **Backfill ETA.** ~3.5 s per bird with at most 50 rows per 5-min sweep, so the bound is the sweep count: 141 birds take three sweeps, ≈ 15–20 min (not ~7 min for ~119).
- **Grid** visit sizes come from one `GROUP BY` per page (`store.visit_sizes`), not one `COUNT` per visit.
