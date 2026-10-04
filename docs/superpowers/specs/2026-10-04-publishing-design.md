# Birdy — Publishing Design

**Date:** 2026-10-04
**Status:** approved for implementation
**Sub-project:** 3 (follows Telegram alerts 6)
**Depends on:** `master` at the Telegram merge (`7d27f6c`).

## Goal

A public gallery at **https://birdy.cartaveo.com** showing the best bird photos from the feeder, kept up to date by the Pi without anyone curating it daily, with manual overrides for the photos the rules get wrong. The homepage shows today's visitors per collaborator side by side; each collaborator has a dashboard with their statistics; each species has a page across all collaborators. The format is multi-site from day one so that sub-project 4 (friends' Pis) is "add another site", not a new system.

## Decisions made

| Question | Decision |
|---|---|
| Target | The user's Hetzner server (`91.98.229.209`, Ubuntu 26.04). It already runs the **cartaveo** Docker Compose stack whose Caddy container (`cartaveo-web-1`) owns ports 80/443. The gallery is a separate folder served by that Caddy under a new site block. Nothing in cartaveo changes otherwise. Combell is not used. |
| Hostname | `birdy.cartaveo.com` (A record → `91.98.229.209`; Caddy obtains the certificate). |
| What is published | Rule-based with manual overrides: a bird photo with a confident species, among the best `per_species_per_day` of that species that local day; `forced`/`hidden` per photo from the review page override the rule. |
| Audience | Public, no auth, exact local capture times shown. |
| Composition | Static files only on the server. Each Pi pushes its own folder (web-sized photos, thumbs, `manifest.json`) with rsync over SSH. Fixed pages render everything in the browser from `sites.json` plus the manifests. No code runs on the server. |
| Pages | Home (today's visitors, one column per site), site dashboard (number cards, charts, species list, grid), species page (across sites). |
| Photo size | Web copy max 1600 px on the long side, JPEG quality 85 (≈300 KB), made with OpenCV (already on the Pi; no new dependency). Thumbs reuse the existing 320 px thumbs. Full-res originals never leave the Pi. |
| Server access | A dedicated `birdy` Unix user on the server; one SSH key per Pi, restricted with `rrsync` to that Pi's `sites/<site_id>/` folder. The Pi never uses root. |

## Architecture

```
Pi (per site)                                              Hetzner (Caddy, static files)
pipeline.process / _backfill_species                        /srv/birdy/
   └─▶ publish.Policy.evaluate(day, species) ─▶ store        ├── index.html  site.html  species.html  app.js  style.css   (from repo gallery/, deployed by deploy/gallery.sh)
            │ published set changed → publisher.notify()    ├── sites.json           (hand-edited list of collaborators)
            ▼                                               └── sites/<site_id>/
Publisher thread (debounce 120 s, max every 900 s):              ├── manifest.json
   sync mirror (resize new, delete unpublished)                   ├── photos/YYYY/MM/DD/HHMMSS_mmm.jpg   (≤1600 px)
   write manifest.json                                            └── thumbs/YYYY/MM/DD/HHMMSS_mmm.jpg   (320 px)
   rsync -az --delete data/publish/ ──▶ birdy@host:<remote_dir>   (rrsync pins the key to sites/<site_id>/)
web POST /captures/{id}/publish ─▶ store.set_published ─▶ publisher.notify()
```

Thread count goes from six to seven when `publish.enabled`.

## Pi side

### Store (`birdy/store.py`)

Migration (same pattern as `species_checked`): `ALTER TABLE captures ADD COLUMN published TEXT` and `ADD COLUMN web_path TEXT`, plus `CREATE INDEX IF NOT EXISTS captures_published ON captures(published)`.

- `published`: `NULL` (not published), `'auto'` (by rule), `'forced'` (manual, always up), `'hidden'` (manual, never up). The published set is `published IN ('auto', 'forced')`.
- `web_path`: relative path of the resized copy under `data/publish/photos/`, `NULL` until the publisher has made it. Same relative path as `path`.
- `Capture` gains `published: str | None` and `web_path: str | None`.

New methods:

```python
def set_published(self, capture_id: int, state: str | None) -> None            # state in (None, "auto", "forced", "hidden")
def set_web_path(self, capture_id: int, web_path: str | None) -> None
def species_day_candidates(self, species: str, day_start_utc: str, day_end_utc: str) -> list[Capture]
    # birds with that species in [day_start, day_end), status 'done', ordered by species_confidence DESC, id ASC
def published(self) -> list[Capture]                                            # published IN ('auto','forced'), newest first
def manual_states(self) -> dict[int, str]                                       # {id: 'forced'|'hidden'}
def publish_stats(self, now: datetime, days: int = 30) -> dict                  # see Manifest.stats; counts ALL birds (is_bird = 1), published or not
```

`purge_non_birds` is unaffected: published photos are birds and birds are never auto-deleted.

### Policy (`birdy/publish.py`, pure)

```python
@dataclass(frozen=True)
class PublishSettings:          # live-tunable subset, from live.get().publish
    min_confidence: float
    per_species_per_day: int

def rank_day(candidates: list[Capture], manual: dict[int, str], settings: PublishSettings) -> dict[int, str | None]
    # returns the desired `published` value for EVERY candidate:
    #   'hidden' stays 'hidden'; 'forced' stays 'forced'
    #   of the rest, those with species_confidence >= min_confidence, ordered by (species_confidence desc, id asc),
    #   the first per_species_per_day get 'auto', the others None
    # forced photos do not consume slots (so a day can show forced + N auto)

class Publisher:  # see below; owns evaluate()
```

`Publisher.evaluate(capture: Capture)` (called by the pipeline): if `capture.is_bird and capture.species`, compute the local day of `capture.captured_at` (via `.astimezone()`), query `species_day_candidates` for that species and day, run `rank_day` with `manual_states()`, apply every changed state with `set_published`, and call `notify()` if anything changed. A capture without a species is never auto-published (it can still be `forced`).

Local day boundaries: `day_start = captured_at.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)`, `day_end = day_start + 1 day`, both converted to UTC strings with the store's `_to_db_time`. Cheap: one indexed query per bird capture.

### Pipeline

- `Pipeline(store, classifier, live, notifier=None, publisher=None, ...)`.
- `process()`: after `update_classification` (and after `notifier.consider`), `if self._publisher: self._publisher.evaluate(self._store.get(capture_id))`. Errors are logged and never mark the row `error` (same contract as `consider`).
- `_backfill_species`: after a row gains a species, `evaluate` it too (unlike alerts, backfilled photos may be published; they're historical, not "events").

### Publisher thread (`birdy/publish.py`)

```python
class Publisher:
    def __init__(self, live: LiveConfig, store: Store, mirror_dir: Path, runner: Callable[[list[str]], int] | None = None,
                 debounce: float = 120.0, max_interval: float = 900.0) -> None
    def evaluate(self, capture: Capture) -> bool          # True if the published set changed
    def set_manual(self, capture_id: int, state: str | None) -> None   # 'forced' | 'hidden' | None (back to rule), then re-evaluate that photo's day
    def notify(self) -> None                             # wake the thread (debounced)
    def run_once(self, now: datetime | None = None) -> bool   # sync mirror + manifest + rsync; True on success (used by tests and `birdy publish`)
    def start() / stop(timeout=5)
```

`run_once`:

1. **Sync mirror** `data/publish/`: for each capture in `store.published()` without a `web_path` (or whose file is missing), read the original with `cv2.imread`, resize so the long side is ≤ 1600 px (`INTER_AREA`, never upscale), write `data/publish/photos/<path>` at JPEG quality 85, copy the existing thumb to `data/publish/thumbs/<path>`, `set_web_path`. If the original can't be read (zero-byte file from a power cut, deleted by hand): `set_published(id, 'hidden')`, one warning, never retried. Then delete any file under `data/publish/photos|thumbs` whose capture is no longer in the published set (and clear its `web_path`).
2. **Write `manifest.json`** (below) atomically (`tmp` + `os.replace`).
3. **rsync**: `rsync -az --delete -e "ssh -i <ssh_key> -o BatchMode=yes -o StrictHostKeyChecking=accept-new" data/publish/ <user>@<host>:<remote_dir>` through `runner` (default `subprocess.call` with a 300 s timeout). Non-zero exit → one warning with the exit code, return False; the mirror is still correct and the next run retries. `remote_dir` defaults to `/`: the server-side `rrsync` restriction roots the key at `sites/<site_id>/`, so the Pi never names the server path itself.

Thread loop: wait on an `Event` with timeout = time until `max_interval` since the last run; when woken by `notify()`, wait a further `debounce` seconds (coalescing bursts, e.g. a backfill sweep publishing 20 photos) before running. `stop()` sets the stop flag and interrupts the wait; a run in progress finishes its rsync (bounded by the 300 s timeout) — `cmd_run` stops the publisher after the pipeline so the last evaluations are included.

### Manifest (`sites/<site_id>/manifest.json`)

```json
{
  "version": 1,
  "generated_at": "2026-10-04T09:12:33+01:00",
  "site": {"id": "home", "name": "home", "lat": 38.8, "lon": -9.39, "since": "2026-10-03", "tz": "Europe/Lisbon"},
  "photos": [
    {"id": 1033, "captured_at": "2026-10-03T11:48:36+01:00", "species": "Common Hoopoe", "confidence": 0.93,
     "photo": "photos/2026/10/03/114836_627.jpg", "thumb": "thumbs/2026/10/03/114836_627.jpg", "w": 1600, "h": 1200, "state": "auto"}
  ],
  "stats": {
    "total_birds": 412,
    "species_count": 7,
    "first_capture": "2026-10-03",
    "last_capture_at": "2026-10-04T08:59:01+01:00",
    "species": [{"name": "Common Hoopoe", "count": 120, "first": "2026-10-03", "last": "2026-10-04T08:59:01+01:00"}],
    "per_day": [{"day": "2026-09-05", "birds": 0}, …, {"day": "2026-10-04", "birds": 37}],
    "per_hour": [0, 0, 0, 0, 0, 0, 2, 14, 31, …]
  }
}
```

- `captured_at` and `last` are local time with offset (the Pi's zone); `per_day` is 30 entries ending today (local days); `per_hour` has 24 entries (local hour of day, all time). `lat`/`lon` rounded to 2 decimals (≈1 km), `null` when unset. `since` is the day of the first capture. `photos` is sorted newest first and contains only the published set.
- Species names with `/` or quotes are plain JSON strings; the page URL-encodes them.

### Config (`[publish]`, `birdy.toml.example`)

```toml
[publish]
enabled = false                   # restart-only; when false the publisher is not constructed
host = ""                         # restart-only; the gallery server, e.g. "91.98.229.209" or "birdy.cartaveo.com"
user = "birdy"                    # restart-only; SSH user on the server
remote_dir = "/"                  # restart-only; rsync target path as the server key sees it ("/" with the rrsync-restricted key;
                                  # the full path, e.g. "/srv/birdy/sites/home", only if the key is unrestricted)
site_id = "home"                  # restart-only; this site's id: its folder name under sites/ and its id in sites.json; [a-z0-9-]{1,32}
ssh_key = "~/.ssh/birdy_publish"  # restart-only; private key for the publish user (install.sh generates it)
min_confidence = 0.6              # live; species_confidence needed to be auto-published (0–1)
per_species_per_day = 3           # live; best N photos per species per local day (1–20)
```

`PublishConfig(enabled, host, user, remote_dir, site_id, ssh_key, min_confidence, per_species_per_day)`. Validation: if `enabled`, `host`, `user`, `remote_dir`, `site_id` non-empty; `site_id` matches `^[a-z0-9-]{1,32}$`; `0 <= min_confidence <= 1`; `1 <= per_species_per_day <= 20`; `ssh_key` expanded with `~`. `ConfigWatcher`'s restart-only set gains the six restart-only keys. `config_to_dict` writes the whole section.

### Settings page

New group **Publishing**: `min_confidence` (number 0–1 step 0.05) and `per_species_per_day` (number 1–20). `SettingsPatch`/`FIELD_KEYS`/`parse_settings_form`/`apply_patch`/template rows as for every tunable. Read-only block: `publish.enabled`, `host`, `user`, `remote_dir`, `site_id`. Changing `min_confidence`/`per_species_per_day` live does not re-rank history; it applies to days evaluated from then on (a `birdy publish --reevaluate` is out of scope; note it in the settings hint).

### Review page (`/`)

- Each card shows a badge: **published** (auto), **forced**, **hidden**, or nothing.
- Two buttons per card: **Publish** (sets `forced`) and **Hide** (sets `hidden`); whichever is current is replaced by **Auto** (back to the rule). Implemented as `POST /captures/{id}/publish` with form field `state=forced|hidden|auto` → `publisher.set_manual(id, state or None)` → 303 back to the same grid page (query string preserved). 404 unknown id, 503 when publishing is disabled (buttons are not rendered then). Still no auth/CSRF: LAN only, like `POST /settings`.
- A `published` filter in the grid dropdown (`filter=published`, shows the published set).

### CLI

- `cmd_run`: if `config.publish.enabled`, build `Publisher(live, store, data_dir / "publish")`, start it, pass it to `Pipeline(...)` and `create_app(..., publisher=publisher)`; stop in `finally` after the pipeline and before the store closes.
- `birdy publish`: run `run_once()` now and exit 0/1 (for first-time setup and debugging). Works while the service runs (WAL, locked connection per process — same caveat as `purge`).
- `deploy/install.sh`: `[ -f ~/.ssh/birdy_publish ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/birdy_publish -C "birdy@$(hostname)"`, then print the public key with the instruction to add it on the server.

## Server side

### Layout

```
/srv/birdy/
├── index.html  site.html  species.html  app.js  style.css   # from repo gallery/ (owner root, world-readable)
├── sites.json                                              # [{"id": "home", "name": "Yassine", "place": "Sintra, PT"}]
└── sites/<site_id>/                                        # owned by birdy:birdy; rsync target
```

- `deploy/gallery.sh` on the Mac: `rsync -az gallery/ root@91.98.229.209:/srv/birdy/` (pages only, excludes `sites/` and `sites.json`). Run when the pages change.
- `sites.json` is hand-edited on the server; it is the only per-collaborator onboarding step besides the key.

### Caddy

One site block added to the cartaveo Caddyfile and `/srv/birdy` bind-mounted read-only into the `web` service (`- /srv/birdy:/srv/birdy:ro` in its `volumes`). The user applies these two edits and reloads (`docker compose up -d web` or `docker exec cartaveo-web-1 caddy reload --config …`): the spec deliberately keeps Claude out of the production stack's files.

```
birdy.cartaveo.com {
    root * /srv/birdy
    file_server
    encode gzip
    header /sites/* Cache-Control "public, max-age=300"
    header /*.json Cache-Control "no-cache"
}
```

### Publish user and key restriction

```bash
adduser --system --group --home /srv/birdy/sites --shell /usr/sbin/nologin birdy   # or useradd equivalent
mkdir -p /srv/birdy/sites/home && chown -R birdy:birdy /srv/birdy/sites
# /home/birdy/.ssh/authorized_keys (or /srv/birdy/sites/.ssh/…), one line per Pi:
command="/usr/bin/rrsync /srv/birdy/sites/home",restrict ssh-ed25519 AAAA… birdy@birdy
```

`rrsync` (ships with rsync on Ubuntu, `/usr/bin/rrsync`) confines the key to that folder: every path the client sends is taken relative to it, so the Pi's target `birdy@host:/` lands in `/srv/birdy/sites/home/` and the key cannot reach any other site. `restrict` disables forwarding and ptys. The exact `authorized_keys` line is verified on the server during the Pi/server task and copied into CLAUDE.md.

### Gallery pages (`gallery/`, vanilla HTML/CSS/JS, no build step, no dependencies)

- `app.js` loads `sites.json`, then every `sites/<id>/manifest.json` in parallel (`fetch`, `cache: "no-cache"`); a site whose manifest fails shows an "unavailable" card instead of breaking the page.
- **Home** `index.html`: a row of columns, one per site in `sites.json` order. Column header: name, place, link to the dashboard. Body: **today's visitors** — one card per species seen today. "Today" is the *site's* local day, not the viewer's: each photo's `captured_at` carries the site's UTC offset, so its local date is read straight from the string (`YYYY-MM-DD` prefix) and compared with the site's current local date (`now` shifted by that offset). Card: best photo of the day (highest confidence), species name, and the number of published photos of it today. Newest species first. Empty day: "No visitors yet today" and the previous active day's cards at 50% opacity with its date.
- **Site dashboard** `site.html?id=home`: header (name, place, "watching since", last visit), number cards (species seen, birds today, birds last 7 days, total birds — from `stats`), charts drawn as inline SVG bars (birds per day, 30 days; visits by hour of day), top species table (name, count, first, last; names link to the species page), then the full published grid newest first with a species filter.
- **Species** `species.html?name=Common%20Hoopoe`: name, per-site cards (site, count, first, last), then all published photos of that species across sites, newest first.
- Photo click opens the web-size image in a minimal lightbox (`<dialog>`); thumbs are `loading="lazy"`.
- Mobile first: columns stack under 700 px.
- Styling: light/dark via `prefers-color-scheme`, system font, no framework.

## Error handling

| Case | Behaviour |
|---|---|
| Server unreachable / rsync fails | one warning with the exit code, retry at the next cycle (≤ 15 min); mirror intact |
| Original unreadable (0-byte, deleted) | capture set `hidden`, one warning, not retried |
| `enabled` with missing host/user/site_id or bad `site_id` | `ConfigError` at startup (exit 2) and on settings save |
| SSH key missing | rsync exits 255 → warning each cycle; `birdy publish` prints the key path hint |
| Manual toggle while the thread runs | state change applies in the DB immediately; the next run reconciles the mirror |
| Backfill publishes many old photos at once | debounce coalesces into one run; resize is ~0.1 s per photo on the Pi |
| Manifest missing/invalid on the server | that site's column shows "unavailable"; the others render |
| Restart | nothing lost: state is in the DB and the mirror folder; first run after start reconciles |

## Testing (Mac)

- **Policy** `rank_day`: threshold; top-N by confidence then id; weaker `auto` demoted when a better photo arrives; `forced` kept and not counting; `hidden` kept; no species → never auto; N = 1 and N = 20 bounds.
- **Store**: migration adds both columns on an old DB; `set_published`/`manual_states`/`published()` round-trip; `species_day_candidates` honours the UTC window and status; `publish_stats` numbers on a seeded DB (per_day length 30, per_hour length 24, species first/last, totals count unpublished birds too).
- **Publisher** with a fake `runner` and a temp data dir: `run_once` resizes (long side ≤ 1600, never upscales a small fixture), copies the thumb, sets `web_path`; unpublishing removes the files and clears `web_path`; unreadable original → `hidden`; rsync args (`--delete`, `-e` with the key, `src/` trailing slash, `user@host:root/sites/id/`), failure returns False and logs once; manifest JSON shape and sorting; `evaluate` changes states and returns True only when something changed; `set_manual` re-ranks; debounce/`max_interval` loop with an injected clock (`notify` twice → one run).
- **Pipeline**: live `process` with a species → `evaluate` called; backfill labelling → `evaluate` called; an `evaluate` exception never marks the row `error`.
- **Config/settings**: defaults; `enabled` without host → `ConfigError`; `site_id` regex; ranges; coverage tests; patch round-trip for the two live keys; restart-only keys preserved on save.
- **Web**: badge and buttons render only when publishing is enabled; `POST /captures/{id}/publish` 303 + state change; 404; 503 when disabled; `filter=published`.
- **Gallery**: a Python test parses `gallery/*.html` and checks every referenced local asset exists and that `app.js` contains no `fetch` of anything outside `sites.json`/`sites/`. Manual: `python -m http.server` in a folder with a seeded manifest, open the three pages in a browser (desktop + phone width).
- **Pi / server**: `deploy/deploy.sh`; `install.sh` makes the key; user creates the `birdy` user, authorized_keys line, `sites.json`, Caddy block, DNS; `birdy publish` → files appear in `/srv/birdy/sites/home/`; open `https://birdy.cartaveo.com`; hide a photo on the review page → gone from the gallery within ~2 minutes.

## Out of scope

Friends' onboarding beyond the two documented server steps (sub-project 4 covers inviting them and any shared conventions), comments/likes, RSS, re-ranking history when thresholds change, server-side rendering, publishing the full-resolution originals, Combell.
