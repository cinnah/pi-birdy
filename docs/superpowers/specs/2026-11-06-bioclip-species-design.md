# BioCLIP 2 species recognition — design

Date: 2026-11-06. Replaces the species model of `2026-10-03-species-recognition-design.md` and builds Part 4 (per-site species list) of `2026-10-05-recognition-improvements-design.md`. Assumes the detector gate (`2026-11-06-detector-gate-design.md`) is deployed.

## Why

Measured on 2026-11-06 over 117 hand-labelled bird photos from the Pi (almost all Black Redstart, both sexes, plus a few Robin, Blackcap, Sardinian Warbler):

| Model | Black Redstart named | p ≥ 0.5 | Pi 4 per photo | RAM |
|---|---|---|---|---|
| Coral iNat-2017 MobileNetV2 (current) | 13 of 117 (median rank 41) | 22, mostly wrong American names | 0.1 s | — |
| same + region mask (Part 4 as specced) | 31 | 7 | 0.1 s | — |
| BioCLIP 1 int8 | 38 | 84, wrong on females | 0.9 s | 262 MB |
| **BioCLIP 2** (ViT-L/14, int8 ONNX) | **99** | **100** | 3.5 s | 641 MB |

BioCLIP 2 (imageomics/bioclip-2, MIT) is zero-shot: it scores an image against text embeddings of species names, so the candidate set *is* the region filter and the 2017 taxonomy-matching problem disappears. Its 18 non-redstart answers are plausible (Robin, Blackcap) or below 0.5.

One limit, also measured: on the 524 non-bird crops it names a species at p ≥ 0.5 on 72 (14 %), and decoy prompts ("empty feeder", "flower pot", "leaves", "person") catch only 17 of them. A closed list has no "nothing here" answer. Therefore the species model runs **only on photos the gate or detector already called a bird**, and species rescue is removed.

## Scope

In: a Mac-built int8 ONNX image tower + a global bank of bird-name embeddings; `BioClipSpecies` replacing `InatBirdClassifier`; species only for birds; rescue removed; the per-site species list (fetch, refresh, include/exclude, CLI, settings page); renaming history once with publish slots moving correctly; `onnxruntime` as a dependency; docs. Out: visits/voting, corrections beyond today's Not-a-bird, training, audio, a text tower on the Pi (names are embedded on the Mac).

## Artefacts built on the Mac — `deploy/build-species-model.sh`

Runs once on the Mac (needs ~3 GB of downloads and torch; the Pi never needs torch):

1. A scratch venv (`.build-venv/`, gitignored) with `torch`, `open_clip_torch`, `onnx`, `onnxruntime`.
2. Export `hf-hub:imageomics/bioclip-2`'s image tower with `torch.onnx.export` (opset 17, input `pixels` float32 `[1,3,224,224]`, output `embedding` `[1,768]`), then `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)` → `models/bioclip2-vision-int8.onnx` (~292 MB). Verified 2026-11-06: the int8 export agrees with fp32 on 109/117 top-1; 3.5 s/photo and 641 MB RSS on the Pi 4 with 4 threads (the fp16 community export is 7.6 s / 1.85 GB — not used).
3. The **bank**: every bird species in iNaturalist's taxonomy, fetched from `GET https://api.inaturalist.org/v1/taxa?taxon_id=3&rank=species&is_active=true&per_page=200&order_by=id&id_above=<last id>` (the API caps `page × per_page` at 10,000; `id_above` paging has no cap), ≤ 1 request/s, `User-Agent: birdy (+https://birdy.akibeo.com)`, ~55 requests, ~10,900 species. For each: `id`, `latin = name`, `common = preferred_common_name` (English; may be missing). Prompt `a photo of {latin} ({common}).`, or `a photo of {latin}.` without a common name — the form measured above. Encoded with the BioCLIP 2 text tower, L2-normalised, stored as `models/bioclip2-birds.npz`: `ids` int32, `latin` and `common` str arrays, `emb` float16 `[N,768]`, `logit_scale` float (100.0), `model` str (`"bioclip-2/int8/<first 12 hex of the onnx sha256>"`), `prompt` str (the template). ~17 MB.

Both files are **gitignored** (GitHub's 100 MB limit); `models/README.md` documents them and the script; `deploy/deploy.sh` already rsyncs `models/`, so the first deploy after building pushes ~310 MB once. `tests/` never need them: real-model tests skip when absent, as for the other models.

## Local species list — `birdy/species_list.py`

As Part 4 specified, with the label-matching part dropped (the bank shares iNaturalist's ids):

- **Fetch** `GET https://api.inaturalist.org/v1/observations/species_counts?lat=…&lng=…&radius=<km>&taxon_id=3&quality_grade=research&locale=en&per_page=500&page=N` until exhausted, ≤ 1 request/s, 20 s timeout, same User-Agent. Keep `taxon.rank == "species"` with `count ≥ 3`: `{id: taxon.id, latin: taxon.name, common: taxon.preferred_common_name or latin, count}`. Fewer than 20 → retry once at 4× radius. Verified 2026-11-06: Sintra at 50 km returns 416 results, 354 species.
- **Store** atomically at `data/species/local.json`: `{fetched_at, lat, lon, radius_km, species: [...]}`.
- **When**: a one-shot background thread at service start fetches when the file is missing, older than 90 days, or its lat/lon/radius differ from the config. Failure: one warning, keep the existing file (or none). `birdy species-list [--refresh]` prints the list (count, radius, age, species not in the bank — expected to be none) and `--refresh` fetches now.
- **No `site.lat/lon`** → no list: the candidate set is the whole bank (10,900 species) and `/settings` shows "set a location to enable the local species list". It still works, just with more lookalikes available.
- **Candidates** = bank rows whose `id` is on the list, plus `species.include`, minus `species.exclude`. `include`/`exclude` are lists of names matched casefold against bank `latin` or `common`; an unknown name is rejected by the settings page with the name in the error (and by `load_config` with a warning, not an error, so an edited `birdy.toml` still loads). Live: a settings save or watcher reload recomputes the candidate set (`BioClipSpecies.set_candidates`), and so does a finished list refresh.

## Classifier — `birdy/species.py` becomes `BioClipSpecies`

```python
class BioClipSpecies:
    def __init__(self, model_path: Path, bank_path: Path, threads: int = 3) -> None: ...
    model_id: str                      # bank["model"]; must equal the onnx's id, else ValueError at load
    def set_candidates(self, ids: Iterable[int] | None) -> int: ...   # None = whole bank; returns the count
    def top(self, img_rgb: np.ndarray, k: int = 5) -> list[tuple[SpeciesLabel, float]]: ...
```

- `SpeciesLabel(latin, common)` stays (the CLI, the upload box and the tests use it). `parse_label` and the iNat label file go.
- Preprocess: short side to 224 (`INTER_AREA` when shrinking, `INTER_CUBIC` when enlarging; agrees with the PIL run on 106/117 crops), centre crop, CLIP mean/std, NCHW float32 — what was measured; implemented with OpenCV on the RGB array the `CropLoader` already produced.
- `top`: one `onnxruntime` run (`intra_op_num_threads = 3`, leaving a core for the capture loop; measured 3.5 s at 4 — the plan re-measures at 3), L2-normalise, `logits = logit_scale · E_candidates · e`, softmax over the **candidates only**, best `k`. Thread safety as for the other models: called under `TwoStageClassifier._lock`.
- `TwoStageClassifier.classify_image`: species runs **only when `is_bird`** (gate or detector); otherwise `top = []`, `species = None`. The rescue rule, `species.rescue_threshold`, its settings row, `FIELD_KEYS` entry, config validation and tests are removed. `confidence` = `max(p_gate, p_det)` always. A species-model failure still degrades to "bird, no species", logged once per distinct error.
- Reported species = top-1 `common` when `p ≥ species.threshold` (0.5 unchanged; names 100 of 117 birds).

## Config

```toml
[species]
model = "models/bioclip2-vision-int8.onnx"   # restart-only
bank = "models/bioclip2-birds.npz"           # restart-only
threshold = 0.5            # name a bird at or above this (live)
region_radius_km = 50      # iNaturalist radius for the local species list; a change refetches at the next start
include = []               # names to add to the local list, latin or common (live)
exclude = []               # names to drop from it (live)
```

`labels` and `rescue_threshold` are removed from `SpeciesConfig`; `load_config` ignores unknown keys (the Pi's `birdy.toml` still has `rescue_threshold = 1.0`; a settings-page save drops it). `region_radius_km` validated 1–500; `include`/`exclude` lists of non-empty strings. `config_to_dict`, `SettingsPatch`, `apply_patch`, `parse_settings_form` and the settings page gain `species_include`/`species_exclude` (textarea, one name per line, like the Telegram watchlist) and `region_radius_km` (restart-only, shown read-only with a note). `test_config_to_dict_covers_every_field` covers them.

## Runtime wiring

- `pyproject.toml`: `"onnxruntime>=1.20"` (aarch64 cp313 wheels exist; installed on the Pi on 2026-11-06). `deploy/install.sh` unchanged.
- `cli._classifier` builds `BioClipSpecies(config.species.model, config.species.bank)`; a missing or mismatched file raises `ModelLoadError` → exit 2. It then applies the current local list (`species_list.load(data_dir)`) + include/exclude via `set_candidates`, and `cmd_run` starts the refresh thread, which calls `set_candidates` again when a fetch lands.
- `birdy classify` prints `species (BioCLIP 2, N candidates):` with the top-5, or `species: skipped (not a bird)`. `POST /classify` shows the same.
- `/species` loses the Part-4 "species the model can't recognise" note (none by construction) and shows the list size and age.
- `/status` unchanged.

## Renaming history

A new `meta` table (`key TEXT PRIMARY KEY, value TEXT`). `cmd_run`, after building the classifier, calls `store.species_model_changed(model_id)`: if `meta.species_model` differs, it sets `species_checked = 0` on every `status = 'done' AND is_bird = 1 AND manual_at IS NULL` row, stores the new id, and logs how many rows will be renamed. The existing backfill (`_backfill`, 50 rows per sweep, yields to live events) then re-runs species on them: a row with `species_checked = 0` always takes the new verdict (the detector-gate rule "a seen row keeps its species" applies only to `species_checked = 1` rows). ~640 rows × 3.5 s ≈ 40 min on the Pi, in 50-row slices of ~3 min each inside the 5-min sweep.

Publishing: when `_backfill` changes a bird's species (old ≠ new, including old → None), it first calls `publisher.forget(old_row)` (re-ranks the old species/day without this row, so the slot moves to the next-best photo) and then `evaluate(new_row)` as today. The web mirror and gallery follow on the next publish run. Alerts: the backfill never alerts; after renaming, a live Black Redstart is not "new" (the DB has many).

Watchlist: names must match the bank's common names (iNaturalist English: "Common Hoopoe", "Eurasian Blue Tit"). The settings page validates the watchlist against the bank and shows unknown names as an error; `AlertPolicy` matching stays casefold.

## Pi operations

- CPU: a bird photo now costs ~4 s of classification (gate + detector + BioCLIP 2) instead of ~0.4 s; empty frames cost the same as today. A visit of 10 photos queues for ~40 s; the queue holds 50 and the cooldown is 3 s. The upload box waits at most one classification.
- Memory: +~650 MB (service ≈ 1 GB total; the Pi has 3.8 GB).
- First deploy: ~310 MB rsync; first start logs the rename count; the gallery moves slots over the next hour.
- Browser note (unrelated bug found the same day): Chrome on the Mac cannot open `birdy.local` because avahi advertises an IPv6 link-local address first; `use-ipv6=no` in `/etc/avahi/avahi-daemon.conf` on the Pi fixes it. Goes in CLAUDE.md.

## Testing (Mac)

- `BioClipSpecies` with a **stand-in ONNX** built in the test (an `onnx.helper` graph: `pixels → Flatten → MatMul(const) → embedding`, deterministic) and a tiny bank fixture (3 species): candidates None/subset/empty, softmax over candidates only, `top` order and k, `SpeciesLabel` shape, model-id mismatch raises, preprocessing shape/dtype. `onnx` becomes a dev extra.
- Real model (skipped when absent): `tests/fixtures/bird.jpg` yields a confident bird species from the whole bank.
- `TwoStageClassifier`: species not called for non-birds (fake species counts calls), no rescue, confidence = max(gate, det), live threshold.
- `species_list`: parse a saved `species_counts` response (fixture with a hybrid and a `count < 3`), radius retry, refetch conditions (missing / 90 days / changed lat-lon-radius), failure keeps the old file, include/exclude matching and unknown-name errors, `set_candidates` called on refresh.
- Store: `meta` table, `species_model_changed` resets only uncorrected done birds and is idempotent.
- Pipeline: a renamed published row triggers `forget(old)` then `evaluate(new)`; a `species_checked = 0` row always takes the new name; a `species_checked = 1` row keeps it.
- Settings: include/exclude/radius round trip, watchlist validation against the bank, `rescue_threshold` gone from the form and the config.
- CLI: `species-list` output, `classify` prints the BioCLIP block / "skipped".
