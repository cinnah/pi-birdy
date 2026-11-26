# Birdy — Telegram Alerts Design

**Date:** 2026-10-03
**Status:** approved for implementation
**Sub-project:** 6 (follows species recognition 2)
**Depends on:** `master` at the species merge (`304389a`).

## Goal

> **Changed 2026-11-26:** the watchlist and `alert_new_species` are gone. Every visit with a species alerts (reason `"visit"`, or `"new"` for a first-ever species, which skips the cooldown) unless the species is on `telegram.no_alert`; a listed species never alerts. The old keys are ignored on load. The settings group is now the **Messages** part and the button reads **Send test message**. The rest of this document describes the original watchlist design.

Get a Telegram message with the photo when a bird you care about visits: species on a watchlist, and the first time any species is ever seen. Bounded per-species frequency, never interferes with capture or classification, configurable from the settings page, verifiable with a test button.

## Decisions made

| Question | Decision |
|---|---|
| Bot | The user's own bot via @BotFather (created: `@Birdyalert_bot`); token and chat id live only in `birdy.toml` on the Pi. |
| Trigger | Watchlist species **or** first-ever sighting of a species (`alert_new_species`). Only confident species (`pred.species` set) on live captures that are birds; backfilled rows never alert. |
| Frequency | Per-species cooldown `cooldown_minutes` (default 30). A "new species" alert bypasses the cooldown (it fires exactly once by definition). |
| Content | `sendPhoto` with the full-res JPEG; caption `🐦 <species> (<species_confidence>%) · HH:MM` plus ` · new species!` for first sightings, then a link `<base_url>/?species=<urlencoded>`. |
| Wiring | `TelegramNotifier` with a queue + one worker thread, fed by the pipeline right after it stores a live classification. No new dependency (`urllib`). |
| Config | `[telegram]` section; `enabled`, `bot_token`, `chat_id`, `base_url` restart-only; `watchlist`, `alert_new_species`, `cooldown_minutes` live-tunable on the settings page. |
| Test | "Send test alert" button on the settings page (`POST /settings/test-alert`) and a `birdy telegram-chat-id` CLI helper. |

## Architecture

```
pipeline.process() ──after update_classification (live only)──▶ notifier.consider(capture_id, path, pred, now)
                                                                   │ AlertPolicy.decide(species, first_seen, now)
                                                                   ▼ reason ("watchlist" | "new") or None
                                                             queue.Queue(20) ──▶ worker thread ──▶ Telegram API
```

### `birdy/notify.py`

```python
@dataclass(frozen=True)
class TelegramSettings:      # the live-tunable subset, read from live.get().telegram on each decide()
    watchlist: tuple[str, ...]
    alert_new_species: bool
    cooldown_minutes: int

class AlertPolicy:
    """Pure decision logic; no I/O. State: last alert time per species (in memory)."""
    def decide(self, species: str, first_seen: bool, settings: TelegramSettings, now: datetime) -> str | None
    # "new" if first_seen and alert_new_species (bypasses cooldown; records the time)
    # "watchlist" if species matches a watchlist entry case-insensitively and
    #     (no previous alert for that species or now - last >= cooldown); records the time
    # None otherwise

@dataclass(frozen=True)
class Alert:
    capture_id: int
    photo_path: Path
    species: str
    species_confidence: float
    captured_at: datetime
    reason: str  # "watchlist" | "new" | "test"

class TelegramTransport(Protocol):
    def send_photo(self, chat_id: str, photo: Path, caption: str) -> None: ...
    def send_text(self, chat_id: str, text: str) -> None: ...

class HttpTelegramTransport:    # urllib multipart POST to https://api.telegram.org/bot<token>/sendPhoto | sendMessage; 20 s timeout
                                # raises TelegramError(status, description) on non-OK responses

class TelegramNotifier:
    def __init__(self, live: LiveConfig, store: Store, transport: TelegramTransport, policy: AlertPolicy | None = None, maxsize: int = 20, sleep=time.sleep)
    def consider(self, capture_id: int, photo_path: Path, pred: Prediction, captured_at: datetime) -> str | None
        # returns the reason if queued; None if not enabled / no species / not a bird / policy says no / queue full (warn)
        # first_seen = store.count("birds", species=pred.species) == 1  (this capture is the only one)
    def send_test(self) -> None      # synchronous sendMessage "birdy test alert from <site.name>"; raises TelegramError to the caller
    def start() / stop(timeout=5)
    # worker: get(timeout=0.5) → build caption → transport.send_photo; on FileNotFoundError → send_text with the same caption;
    # on TelegramError/OSError → retry after 5 s, 15 s, 45 s (via injected sleep), then log.error once and drop
```

Caption: `f"🐦 {species} ({int(round(p*100))}%) · {captured_at.astimezone():%H:%M}"` + (` · new species!` if reason == "new") + `\n{base_url}/?species={quote(species)}`.

The bot token is read from `live.get().telegram.bot_token` at transport construction (restart-only). It is never logged: `HttpTelegramTransport` formats error messages from Telegram's `description` only, and the URL is never included in exceptions.

### Config (`[telegram]`)

```toml
[telegram]
enabled = false            # restart-only; when false the notifier is not even constructed
bot_token = ""             # restart-only, secret (birdy.toml is gitignored and rsync-excluded)
chat_id = ""               # restart-only
base_url = "http://birdy.local:8000"   # restart-only; link in the caption
watchlist = []             # live-tunable; common names as shown on /species, case-insensitive
alert_new_species = true   # live-tunable
cooldown_minutes = 30      # live-tunable, >= 0
```
`TelegramConfig(enabled, bot_token, chat_id, base_url, watchlist: tuple[str, ...], alert_new_species, cooldown_minutes)`. Validation: `cooldown_minutes >= 0`; if `enabled`, `bot_token` and `chat_id` must be non-empty (`ConfigError` naming `telegram.bot_token` / `telegram.chat_id`); `watchlist` entries must be non-empty strings (trimmed; empty entries dropped). `ConfigWatcher`'s restart-only set gains `telegram.enabled/bot_token/chat_id/base_url`.

### Settings page
- New group **Alerts**: `watchlist` as a `<textarea name="watchlist">` (one species per line; parsed by splitting lines, trimming, dropping empties), `alert_new_species` checkbox (form value present → true), `cooldown_minutes` number (0–1440). Three new `SettingsPatch` fields: `watchlist: tuple[str, ...]`, `alert_new_species: bool`, `cooldown_minutes: int`. `FIELD_KEYS` gains `watchlist → telegram.watchlist`, `alert_new_species → telegram.alert_new_species`, `cooldown_minutes → telegram.cooldown_minutes`.
- Next to the textarea, a hint listing the species seen so far (from `store.species_summary()`), so spellings can be copied.
- Read-only block: `telegram.enabled`, `telegram.chat_id`, `telegram.bot_token: set | not set` (never the value), `telegram.base_url`.
- **Send test alert** button: `POST /settings/test-alert` → `notifier.send_test()`; renders the settings page with a banner "test alert sent" or the Telegram error description (status 502). 503 when alerts are disabled (notifier is None).

### Pipeline
- `Pipeline(store, classifier, live, notifier: TelegramNotifier | None = None, ...)`. In `process()` (live path only), after `update_classification` succeeds: `if self._notifier: self._notifier.consider(capture_id, path, pred, captured_at)`. The pipeline needs `captured_at`: `process(capture_id, path, captured_at)` — `MotionEvent` already carries it; the pending sweep passes `c.captured_at`. `_backfill_species` never calls the notifier.

### CLI
- `cmd_run`: if `config.telegram.enabled`, build `HttpTelegramTransport(token)` + `TelegramNotifier(live, store, transport)`, start it, pass it to `Pipeline(...)` and `create_app(..., notifier=notifier)`; stop it in `finally` (after the pipeline).
- New subcommand `birdy telegram-chat-id`: calls `getUpdates` with `config.telegram.bot_token` (or `--token`) and prints `chat_id  name  last text` for each chat that has messaged the bot; exits 1 with a hint if none ("send your bot a message first").

### Bot setup (documented in CLAUDE.md)
1. Telegram → @BotFather → `/newbot` → copy the token into `[telegram] bot_token` in `~/pi-birdy/birdy.toml` on the Pi.
2. Open the bot and send it any message.
3. `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy telegram-chat-id'` → paste the id into `chat_id`, set `enabled = true`, `sudo systemctl restart birdy`.
4. Settings page → **Send test alert**.

## Error handling

| Case | Behaviour |
|---|---|
| `enabled` but token/chat id empty | `ConfigError` at startup (exit 2) and on settings save |
| Telegram 4xx/5xx or timeout on an alert | retry 5/15/45 s, then `log.error` once and drop; service unaffected |
| 401 (bad token) | same as above; the test button shows Telegram's description inline |
| Queue full | newest alert dropped with a warning |
| Photo missing at send time | text-only alert with the same caption |
| Transport raises unexpectedly | worker logs and continues |
| Restart | cooldown state lost (one possible repeat per species); "new species" state comes from the DB and survives |

## Testing (Mac)
- `AlertPolicy`: watchlist hit; case-insensitive and whitespace-tolerant match; non-watchlist, not new → None; new bypasses cooldown and records time; cooldown suppresses within window and re-allows after; `cooldown_minutes = 0` never suppresses.
- `TelegramNotifier` with a `FakeTransport` (records calls, can raise `TelegramError` N times, can raise `FileNotFoundError`): `consider` returns None when disabled/no species/non-bird/queue full; caption format incl. link and "new species!"; `send_photo` called with the capture path; text fallback when the photo is missing; retry sequence with injected sleep (5, 15, 45) then drop with one error log; worker survives a generic exception; `send_test` text and propagation of `TelegramError`.
- `HttpTelegramTransport`: multipart body construction tested with a fake `urlopen` (asserts the boundary/filename/caption fields and the URL path, never asserting on the token in logs); non-OK JSON → `TelegramError(description)`.
- Config: defaults; `enabled` without token/chat → `ConfigError`; watchlist parsing (list of strings, trimmed); `cooldown_minutes` range; coverage test picks up the new section; `config_to_dict` round-trips the token into the file.
- Settings: textarea → tuple; checkbox absent → False; patch round-trip; restart-only keys preserved on save.
- Web: Alerts group renders with the species hint; read-only block says "set"/"not set" and never contains the token string; `POST /settings/test-alert` → 200 banner on success, 502 with description on `TelegramError`, 503 when the notifier is None.
- Pipeline: live `process` with a species → `consider` called with the right args; backfill never calls it; a `consider` exception is logged and does not mark the row error.
- CLI: `telegram-chat-id` with a fake `getUpdates` response prints ids; none → exit 1.
- Pi: deploy; token + chat id already known (chat id 1115872433); enable; test button; show the phone a hoopoe → photo alert.

## Implementation notes (as merged)

- `TelegramNotifier(..., sleep=None)`: the default sleep is the stop event's `wait`, so `stop()` interrupts a retry back-off. Queued alerts are dropped on shutdown.
- Every transport failure (HTTP error, timeout, DNS, invalid URL, bad JSON) is wrapped as `TelegramError` with the token redacted; `send_test` therefore raises only `TelegramError`.
- Telegram 400/401/403 are not retried: they mean a bad chat id, a revoked token or a blocked bot, which a retry can't fix.
- `Alert.reason` is only ever `"watchlist"` or `"new"`; the test button sends a text message directly and never builds an `Alert`.
- `chat_id` and `base_url` are captured in `TelegramNotifier.__init__` (restart-only, like the token); `enabled` is checked per alert, so `enabled = false` applies live while `enabled = true` needs a restart.

## Out of scope
Multiple recipients/groups, per-species cooldowns, quiet hours (the daylight gate already silences nights), alert history page, inline buttons/commands on the bot, other channels (email, push).
