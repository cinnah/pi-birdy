# Visits and Corrections Implementation Plan

**Status:** implemented on branch `visits`, 2026-11-07. Deviations from the spec are listed in the spec's "Implementation notes / deviations" section.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Group a bird's photos into visits, decide one species per visit from the pooled BioCLIP embedding, alert once per visit, and let a person correct a visit's label (not a bird / species unknown / a species) from the grid, with gallery slots following every change.

**Architecture:** `captures` gains `visit_id` and `embedding`; a `visits` table holds the span and the resolved label. The pipeline writes the classifier's verdict to `model_*` + `embedding` (`store.store_verdict`) and hands the row to `birdy/visits.py::Visits`, which assigns/merges visits, resolves final columns (correction > keep-own > vote > unnamed), moves publish slots through `Publisher.evaluate_day`, closes visits and alerts on a 5 s tick. Corrections go through `Visits.correct`. A one-time migration (guarded by `meta.visits_version`) builds visits over history and queues the embedding backfill.

**Tech Stack:** Python 3.11/3.13, SQLite, numpy, FastAPI/Jinja, pytest. Spec: `docs/superpowers/specs/2026-11-07-visits-and-corrections-design.md`.

**Conventions:** tests with `.venv/bin/pytest`; branch `visits` off master; `git add` by file name; commit after each task ending with
```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Agk9eUVj9dbThFrVMdZie7
```

**One deliberate deviation from the spec, decided here:** `Store.update_classification` keeps writing both `model_*` and the final columns (it is what ~80 existing tests and `resolve_single` need: "store a verdict and resolve as a lone photo"). The pipeline uses the new `store_verdict`, which writes only `model_*`, `embedding`, `det_*`, the checked flags and `status`; final columns for visit members are written only by `Visits.resolve`. Task 10 updates the spec sentence.

---

## File map

| File | Change |
|---|---|
| `birdy/config.py`, `birdy.toml.example`, `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` | `[visits] gap_seconds` (live) |
| `birdy/store.py` | schema (`visits`, `visit_id`, `embedding`), `Capture` fields, `store_verdict`, `write_final`, visit CRUD/queries, `set_manual`, `migrate_visits` |
| `birdy/species.py`, `birdy/classifier.py` | `score_candidates`, `Prediction.embedding`, `top_with_embedding` |
| `birdy/visits.py` (new) | `vote`, `final_label`, `Visits` |
| `birdy/notify.py` | `consider(..., first_seen)` |
| `birdy/publish.py` | `evaluate_day` |
| `birdy/pipeline.py` | `store_verdict` + `Visits`, tick, no per-photo alerts |
| `birdy/retention.py`, `birdy/cli.py` | visit hooks, `Visits` wiring, migration call |
| `birdy/web/app.py`, `birdy/web/templates/index.html`, `birdy/web/static/osd.css` | label endpoint, Fix panel, `?visit=`, badges |
| `CLAUDE.md`, spec | docs |
| tests: `test_config.py`, `test_settings.py`, `test_web.py`, `test_store.py`, `test_species.py`, `test_two_stage.py`, `test_visits.py` (new), `test_notify.py`, `test_publish.py`, `test_pipeline.py`, `test_retention.py`, `test_cli.py` | |

---

### Task 1: `[visits] gap_seconds`

**Files:** `birdy/config.py`, `birdy.toml.example`, `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` (settings values); tests `tests/test_config.py`, `tests/test_settings.py`, `tests/test_web.py`, `tests/test_watcher.py`

- [x] **Step 1: Failing tests**

`tests/test_config.py` append:
```python
def test_visits_defaults_and_range(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.visits.gap_seconds == 60
    for bad in ("gap_seconds = 5", "gap_seconds = 601", 'gap_seconds = "x"'):
        with pytest.raises(ConfigError, match="visits.gap_seconds"):
            load_config(write(tmp_path, MINIMAL + f"\n[visits]\n{bad}\n"))
```
`tests/test_settings.py`: `default_patch` gains `visits_gap_seconds=config.visits.gap_seconds,`; `FORM` gains `"visits_gap_seconds": "60",`; `FULL_TOML` gains `[visits]\ngap_seconds = 90`; append:
```python
def test_visits_gap_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, visits_gap_seconds=120))
    assert new.visits.gap_seconds == 120 and load_config(config_path).visits.gap_seconds == 120
```
`tests/test_web.py` `_base_form`: add `"visits_gap_seconds": "60",`; append `test_settings_page_has_visits_gap(client)` asserting `name="visits_gap_seconds"` in GET /settings. `tests/test_watcher.py::test_save_settings_then_watch_is_noop`: add `visits_gap_seconds=cur.visits.gap_seconds,`.

- [x] **Step 2: Run** `.venv/bin/pytest tests/test_config.py tests/test_settings.py -q` → fails on `cfg.visits`.

- [x] **Step 3: Implement**

`config.py`: `@dataclass(frozen=True) class VisitsConfig: gap_seconds: int  # photos of one bird within this gap form a visit (live)`; `Config.visits: VisitsConfig`; parse `vis = raw.get("visits", {})`, `gap = _num(vis, "visits", "gap_seconds", 60, int)`, `_check(10 <= gap <= 600, "visits.gap_seconds must be between 10 and 600")`; constructor `visits=VisitsConfig(gap_seconds=gap)`.
`birdy.toml.example`: after `[detector]`:
```toml
[visits]
gap_seconds = 60           # photos of one bird within this gap form a visit (10–600); applies without a restart
```
`settings.py`: `FIELD_KEYS["visits_gap_seconds"] = "visits.gap_seconds"`; `SettingsPatch.visits_gap_seconds: int`; `config_to_dict["visits"] = {"gap_seconds": ...}`; `apply_patch`: `data.setdefault("visits", {})["gap_seconds"] = patch.visits_gap_seconds`; `parse_settings_form`: `visits_gap_seconds=_num(form, "visits_gap_seconds", int)`.
`app.py` settings values: `"visits_gap_seconds": cfg.visits.gap_seconds`. `settings.html`: a new section after Bird check:
```jinja
  <section class="group">
    <header><h2 class="head">Visits</h2><p>Photos of one bird become one record</p></header>
    {{ num("visits_gap_seconds", "visits.gap_seconds", "Visit gap",
           "Photos less than this far apart are the same bird visit: one species, one alert, one correction.", 10, 600, 5, "s") }}
  </section>
```

- [x] **Step 4: Run** `.venv/bin/pytest -q` → all pass. **Step 5: Commit** "Config: [visits] gap_seconds".

---

### Task 2: Store — schema, verdict/final split, visit methods

**Files:** `birdy/store.py`; `tests/test_store.py`

- [x] **Step 1: Failing tests** (append to `tests/test_store.py`; `add`, `add_species`, `T0`, `timedelta` exist):

```python
import numpy as np


def test_store_verdict_writes_model_columns_only(store):
    cid = add(store, T0)
    emb = np.full(768, 0.036, np.float32)
    store.store_verdict(cid, is_bird=True, confidence=0.7, species="Robin", species_confidence=0.9,
                        det_confidence=0.6, det_box=(0.1, 0.2, 0.3, 0.4), embedding=emb)
    c = store.get(cid)
    assert c.status == "done" and c.species_checked and c.model_is_bird is True and c.model_confidence == 0.7
    assert c.model_species == "Robin" and c.model_species_confidence == 0.9 and c.det_confidence == 0.6
    assert c.is_bird is None and c.species is None            # final columns untouched
    assert c.has_embedding and np.allclose(store.embedding(cid), emb, atol=1e-3)
    store.store_verdict(cid, is_bird=False, confidence=0.1, species=None, species_confidence=None, embedding=None)
    assert store.embedding(cid) is not None                    # a None embedding keeps the stored one


def test_write_final(store):
    cid = add(store, T0)
    store.write_final(cid, is_bird=True, confidence=0.7, species="Robin", species_confidence=0.8)
    c = store.get(cid)
    assert (c.is_bird, c.confidence, c.species, c.species_confidence) == (True, 0.7, "Robin", 0.8)


def test_visit_crud(store):
    a = add(store, T0); b = add(store, T0 + timedelta(seconds=30)); c = add(store, T0 + timedelta(minutes=10))
    v1 = store.create_visit(T0, T0, alert="pending")
    store.attach_to_visit(a, v1); store.attach_to_visit(b, v1, extend=True)
    v = store.get_visit(v1)
    assert v.started_at == T0 and v.ended_at == T0 + timedelta(seconds=30) and v.closed is False and v.alert == "pending"
    assert [x.id for x in store.visit_photos(v1)] == [a, b]
    assert store.get(a).visit_id == v1
    v2 = store.create_visit(store.get(c).captured_at, store.get(c).captured_at, alert="skipped"); store.attach_to_visit(c, v2)
    assert [x.id for x in store.visits_overlapping(T0 - timedelta(seconds=60), T0 + timedelta(seconds=90))] == [v1]
    assert [x.id for x in store.visits_overlapping(T0 + timedelta(minutes=9), T0 + timedelta(minutes=11))] == [v2]
    store.merge_visits(v1, [v2])
    assert store.get_visit(v2) is None and store.get(c).visit_id == v1 and store.get_visit(v1).ended_at == store.get(c).captured_at
    store.set_visit_result(v1, "Robin", 0.9); store.close_visit(v1); store.set_visit_alert(v1, "sent")
    v = store.get_visit(v1)
    assert (v.species, v.species_confidence, v.closed, v.alert) == ("Robin", 0.9, True, "sent")
    assert store.open_visits() == []
    assert store.count_visits_with_species("Robin") == 1
    store.delete_visit(v1)
    assert store.get_visit(v1) is None and store.get(a).visit_id is None


def test_open_visits_and_members(store):
    a = add(store, T0)
    v = store.create_visit(T0, T0, alert="pending"); store.attach_to_visit(a, v)
    assert [x.id for x in store.open_visits()] == [v]
    store.store_verdict(a, is_bird=True, confidence=0.9, species=None, species_confidence=None)
    store.write_final(a, True, 0.9, None, None)
    assert store.bird_members(v) == [a]
    store.set_manual(a, is_bird=False, species=None)
    assert store.bird_members(v) == [] and store.get(a).manual_is_bird is False and store.get(a).manual_at is not None
    store.set_manual(a, is_bird=True, species="Robin")
    assert store.get(a).manual_species == "Robin" and store.bird_members(v) == [a]
    store.clear_manual(v)
    assert store.get(a).manual_at is None and store.get(a).manual_species is None
```

Also update existing tests that relied on `mark_not_bird` copying `model_*` only on first correction: keep `mark_not_bird`/`clear_correction` working (they stay for the web until Task 8 replaces them).

- [x] **Step 2: Run** → `AttributeError: 'Store' object has no attribute 'store_verdict'` etc.

- [x] **Step 3: Implement** in `birdy/store.py`

Schema additions (`SCHEMA`):
```sql
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL,
  closed INTEGER NOT NULL DEFAULT 0,
  species TEXT,
  species_confidence REAL,
  alert TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS visits_time ON visits(ended_at);
CREATE INDEX IF NOT EXISTS visits_open ON visits(closed, ended_at);
```
ALTER loop: `("visit_id", "INTEGER")`, `("embedding", "BLOB")`, and after the loop `CREATE INDEX IF NOT EXISTS captures_visit ON captures(visit_id)`.

`Capture` gains `visit_id: int | None = None`, `model_confidence: float | None = None` (map `row["model_confidence"]` — it exists in the table already), `manual_species: str | None = None` (map it), `has_embedding: bool = False` (map `row["embedding"] is not None`). Add a `Visit` dataclass: `id, started_at, ended_at, closed: bool, species, species_confidence, alert` with `_row_to_visit`.

Methods:
```python
    def store_verdict(self, capture_id, *, is_bird, confidence, species, species_confidence,
                      det_confidence=0.0, det_box=None, embedding=None) -> None:
        """The classifier's verdict for a photo (model_*); the final columns are written by Visits."""
        bx = (None, None, None, None) if det_box is None else det_box
        blob = None if embedding is None else np.asarray(embedding, np.float16).tobytes()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET model_is_bird = ?, model_confidence = ?, model_species = ?, model_species_confidence = ?,"
                " det_confidence = ?, det_x = ?, det_y = ?, det_w = ?, det_h = ?,"
                " embedding = COALESCE(?, embedding), species_checked = 1, detector_checked = 1, status = 'done' WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, det_confidence, *bx, blob, capture_id),
            )

    def embedding(self, capture_id) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute("SELECT embedding FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return None if row is None or row["embedding"] is None else np.frombuffer(row["embedding"], np.float16).astype(np.float32)

    def embeddings(self, capture_ids) -> dict[int, np.ndarray]: ...  # one SELECT ... WHERE id IN (...) AND embedding IS NOT NULL

    def write_final(self, capture_id, is_bird, confidence, species, species_confidence) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ? WHERE id = ?",
                               (int(is_bird), confidence, species, species_confidence, capture_id))

    def set_manual(self, capture_id, *, is_bird: bool, species: str | None, at: datetime | None = None) -> None:
        """A person's correction on one photo (Visits copies it across the visit)."""
        ...  # UPDATE manual_is_bird, manual_species, manual_at = at or now

    def clear_manual(self, visit_id) -> None: ...   # NULL the manual_* columns on every photo of the visit
    def clear_manual_single(self, capture_id) -> None: ...

    # --- visits ---
    def create_visit(self, started_at, ended_at, alert="pending") -> int
    def get_visit(self, visit_id) -> Visit | None
    def attach_to_visit(self, capture_id, visit_id, extend=False) -> None   # extend: widen started/ended to include the photo
    def visit_photos(self, visit_id) -> list[Capture]                        # ORDER BY captured_at, id
    def bird_members(self, visit_id) -> list[int]                            # ids where manual_is_bird=1 OR (manual_is_bird IS NULL AND (model_is_bird=1 OR is_bird=1))
    def visits_overlapping(self, start, end) -> list[Visit]                  # started_at <= end AND ended_at >= start ORDER BY id
    def merge_visits(self, into, others) -> None                             # move photos, widen span, delete others (one transaction)
    def set_visit_result(self, visit_id, species, species_confidence) -> None
    def close_visit(self, visit_id) -> None
    def set_visit_alert(self, visit_id, state) -> None
    def open_visits(self) -> list[Visit]                                     # closed = 0 ORDER BY ended_at
    def pending_alert_visits(self) -> list[Visit]                            # alert = 'pending' ORDER BY started_at
    def count_visits_with_species(self, species) -> int
    def delete_visit(self, visit_id) -> None                                 # NULL visit_id on its photos, delete the row
    def visit_sizes(self, visit_ids) -> dict[int, int]                       # COUNT(*) GROUP BY visit_id
```
Each is one `with self._lock, self._conn:` transaction, same style as the file. `_row_to_capture` maps the new fields. `update_classification` stays as is **and additionally** copies the verdict into `model_*` (so a lone photo has both): add `model_is_bird = ?, model_confidence = ?, model_species = ?, model_species_confidence = ?` to its UPDATE with the same values. Keep `mark_not_bird`/`clear_correction` for now.

- [x] **Step 4: Run** `.venv/bin/pytest tests/test_store.py -q` → pass; full suite → pass. **Step 5: Commit** "Store: visits table, embeddings, verdict/final split, visit methods".

---

### Task 3: `score_candidates`, `Prediction.embedding`

**Files:** `birdy/species.py`, `birdy/classifier.py`; `tests/test_species.py`, `tests/test_two_stage.py`

- [x] **Step 1: Failing tests**

`tests/test_species.py` append:
```python
def test_score_candidates_matches_top(standin):
    from birdy.species import score_candidates
    sp = BioClipSpecies(*standin)
    img = flat((0, 255, 0))
    top, emb = sp.top_with_embedding(img, k=3)
    assert emb.shape == (D,) and abs(np.linalg.norm(emb) - 1) < 1e-5
    scored = score_candidates(sp.bank, sp.candidate_rows, emb)          # [(bank row, p)] best first
    assert [sp.bank.ids[r] for r, _ in scored[:3]] == [sp.bank.resolve(l.common) for l, _ in top]
    assert abs(scored[0][1] - top[0][1]) < 1e-6
    assert score_candidates(sp.bank, np.array([], np.int64), emb) == []
```
`tests/test_two_stage.py`: `FakeSpecies` gains `top_with_embedding(self, img, k=5)` returning `(self.top(img, k), np.ones(8, np.float32) / np.sqrt(8))`; append:
```python
def test_prediction_carries_the_embedding_for_birds(live):
    pred, _ = two_stage(live, 0.9, 0.8).classify_image(IMG)
    assert pred.embedding is not None and pred.embedding.shape == (8,)
    pred, _ = two_stage(live, 0.1, 0.8, p_det=0.1).classify_image(IMG)
    assert pred.embedding is None
```

- [x] **Step 2: Run** → attribute errors.

- [x] **Step 3: Implement**

`species.py`:
```python
def score_candidates(bank: Bank, cand_rows: np.ndarray, emb: np.ndarray) -> list[tuple[int, float]]:
    """Softmax of logit_scale · E[cand] · emb over the candidate rows; [(bank row, p)] best first."""
    if len(cand_rows) == 0:
        return []
    logits = bank.logit_scale * (bank.emb[cand_rows] @ emb)
    p = np.exp(logits - logits.max()); p /= p.sum()
    order = np.argsort(p)[::-1]
    return [(int(cand_rows[i]), float(p[i])) for i in order]
```
`BioClipSpecies`: `candidate_rows` property (returns `self._cand`), `top_with_embedding(img, k) -> (top, emb)` built on `embed` + `score_candidates`; `top` = `top_with_embedding(...)[0]`.
`classifier.py`: `Prediction.embedding: np.ndarray | None = field(default=None, compare=False, repr=False)`; in `classify_image`, when species runs: `fn = getattr(self.species, "top_with_embedding", None)`; `top, emb = fn(img_rgb, k=5) if fn else (self.species.top(img_rgb, k=5), None)`; pass `embedding=emb`.

- [x] **Step 4: Run** `.venv/bin/pytest tests/test_species.py tests/test_two_stage.py tests/test_cli.py -q` → pass. **Step 5: Commit** "Species: score_candidates; predictions carry the BioCLIP embedding".

---

### Task 4: `birdy/visits.py` — vote, resolve, grouping, tick, corrections

**Files:** create `birdy/visits.py`; `tests/test_visits.py`

- [x] **Step 1: Failing tests** — create `tests/test_visits.py`:

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.species import Bank, load_bank
from birdy.store import Store
from birdy.visits import ALERT_AFTER, KEEP_OWN, Visits, final_label, vote

T0 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
D = 4


def bank(tmp_path) -> Bank:
    # unit vectors: row 0 = Black Redstart (e1), row 1 = Robin (e2), row 2 = Hoopoe (e3)
    p = tmp_path / "bank.npz"
    np.savez(p, ids=np.array([1, 2, 3], np.int32), latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Upupa epops"]),
             common=np.array(["Black Redstart", "European Robin", "Eurasian Hoopoe"]), emb=np.eye(3, D, dtype=np.float16),
             logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return load_bank(p)


def e(i, noise=0.0):
    v = np.eye(D, dtype=np.float32)[i] + noise
    return v / np.linalg.norm(v)


class FakeList:
    version = 1
    def candidates(self, cfg): return None          # whole bank


class FakeNotifier:
    def __init__(self): self.calls = []
    def consider(self, capture_id, photo_path, species, species_confidence, captured_at, first_seen):
        self.calls.append((capture_id, species, first_seen)); return "watchlist"


class FakePublisher:
    def __init__(self): self.days = []
    def evaluate_day(self, species, when): self.days.append((species, when.date()))


@pytest.fixture
def store(tmp_path): return Store(tmp_path / "data")


@pytest.fixture
def visits(store, live, tmp_path):
    return Visits(store, live, bank=bank(tmp_path), species_list=FakeList(), notifier=FakeNotifier(), publisher=FakePublisher())


def add(store, seconds, *, bird=True, species=None, p=None, emb=None, live_verdict=True):
    when = T0 + timedelta(seconds=seconds)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.store_verdict(cid, is_bird=bird, confidence=0.9 if bird else 0.1, species=species, species_confidence=p, embedding=emb)
    return cid


# --- pure functions -------------------------------------------------------

def test_vote_pools_embeddings(tmp_path):
    b = bank(tmp_path); cand = np.arange(3)
    out = vote([e(0), e(0, 0.1), e(1)], b, cand, threshold=0.5)
    assert out is not None and out.species == "Black Redstart" and out.p > 0.5
    assert vote([], b, cand, threshold=0.5) is None
    assert vote([e(0) * 0 + np.full(D, 0.5)], b, cand, threshold=0.99) is None     # flat -> below threshold


def test_final_label_priority():
    assert final_label(manual_is_bird=False, manual_species=None, own=("Robin", 0.9), vote_species="Hoopoe") == (False, None)
    assert final_label(manual_is_bird=True, manual_species="Hoopoe", own=("Robin", 0.9), vote_species="Robin") == (True, "Hoopoe")
    assert final_label(manual_is_bird=True, manual_species=None, own=("Robin", 0.9), vote_species="Robin") == (True, None)
    assert final_label(manual_is_bird=None, manual_species=None, own=("Robin", KEEP_OWN), vote_species="Hoopoe") == (True, "Robin")
    assert final_label(manual_is_bird=None, manual_species=None, own=("Robin", 0.5), vote_species="Hoopoe") == (True, "Hoopoe")
    assert final_label(manual_is_bird=None, manual_species=None, own=None, vote_species=None) == (True, None)


# --- grouping ---------------------------------------------------------------

def test_new_join_and_gap(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 30, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    c = add(store, 200, emb=e(1)); visits.photo_classified(store.get(c), live=True)
    assert store.get(a).visit_id == store.get(b).visit_id != store.get(c).visit_id
    assert store.get(a).species == "Black Redstart" and store.get(b).species == "Black Redstart" and store.get(c).species == "European Robin"
    assert store.get_visit(store.get(a).visit_id).alert == "pending"


def test_backfill_visits_never_alert(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=False)
    assert store.get_visit(store.get(a).visit_id).alert == "skipped"


def test_merge_bridging_photo(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    c = add(store, 100, emb=e(0)); visits.photo_classified(store.get(c), live=True)
    assert store.get(a).visit_id != store.get(c).visit_id
    store.set_visit_alert(store.get(c).visit_id, "sent")
    b = add(store, 50, emb=e(0)); visits.photo_classified(store.get(b), live=True)   # within 60 s of both
    vid = store.get(a).visit_id
    assert store.get(b).visit_id == vid == store.get(c).visit_id and store.get_visit(vid).alert == "sent"
    assert len(store.visit_photos(vid)) == 3


def test_not_bird_visit_is_not_joined(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "not_bird")
    b = add(store, 20, emb=e(1)); visits.photo_classified(store.get(b), live=True)
    assert store.get(b).visit_id != store.get(a).visit_id and store.get(b).manual_at is None


def test_joining_photo_inherits_correction(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:Eurasian Hoopoe")
    b = add(store, 20, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    assert store.get(b).visit_id == store.get(a).visit_id and store.get(b).species == "Eurasian Hoopoe" and store.get(b).manual_species == "Eurasian Hoopoe"


def test_non_bird_gets_no_visit_and_resolve_single(store, visits):
    a = add(store, 0, bird=False, species="Robin", p=0.9); visits.photo_classified(store.get(a), live=True)
    c = store.get(a)
    assert c.visit_id is None and c.is_bird is False and c.species is None and c.model_species == "Robin"


# --- resolve ---------------------------------------------------------------

def test_keep_own_second_bird(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    c = add(store, 20, emb=e(1)); visits.photo_classified(store.get(c), live=True)    # own p ~1.0 for Robin
    assert store.get(c).species == "European Robin" and store.get(a).species == "Black Redstart"


def test_resolve_moves_publish_slots(store, visits):
    pub = visits._publisher
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    store.set_published(a, "auto")
    visits.correct(a, "species:European Robin")
    assert store.get(a).published is None and store.get(a).species == "European Robin"
    assert ("Black Redstart", T0.date()) in pub.days and ("European Robin", T0.date()) in pub.days


def test_species_confidence_is_own_p_or_model_confidence_when_corrected(store, visits):
    a = add(store, 0, emb=e(0, 0.2)); visits.photo_classified(store.get(a), live=True)
    assert 0.5 < store.get(a).species_confidence < 1.0
    visits.correct(a, "species:Eurasian Hoopoe")
    assert store.get(a).species_confidence == store.get(a).model_confidence


# --- tick / alerts -----------------------------------------------------------

def test_tick_closes_and_alerts_once(store, visits):
    n = visits._notifier
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0, 0.3)); visits.photo_classified(store.get(b), live=True)
    vid = store.get(a).visit_id
    visits.tick(T0 + timedelta(seconds=30))
    assert store.get_visit(vid).closed is False and n.calls == []
    visits.tick(T0 + timedelta(seconds=71))
    v = store.get_visit(vid)
    assert v.closed is True and v.alert == "sent" and n.calls == [(a, "Black Redstart", True)]   # best photo = highest own p
    visits.tick(T0 + timedelta(seconds=200))
    assert len(n.calls) == 1


def test_tick_alerts_long_open_visit_early(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    vid = store.get(a).visit_id
    for s in range(50, ALERT_AFTER + 60, 50):
        store.attach_to_visit(add(store, s, emb=e(0)), vid, extend=True)
    visits.tick(T0 + timedelta(seconds=ALERT_AFTER + 1))
    v = store.get_visit(vid)
    assert v.closed is False and v.alert == "sent"


def test_tick_skips_late_and_unnamed(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.tick(T0 + timedelta(minutes=20))
    assert store.get_visit(store.get(a).visit_id).alert == "skipped" and visits._notifier.calls == []
    b = add(store, 3600, emb=np.full(D, 0.5) / np.linalg.norm(np.full(D, 0.5))); visits.photo_classified(store.get(b), live=True)
    visits.tick(T0 + timedelta(seconds=3600 + 61))
    assert store.get_visit(store.get(b).visit_id).alert == "skipped"


def test_first_seen_is_first_visit(store, visits):
    n = visits._notifier
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.tick(T0 + timedelta(seconds=61))
    b = add(store, 600, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    visits.tick(T0 + timedelta(seconds=661))
    assert [c[2] for c in n.calls] == [True, False]


# --- corrections ---------------------------------------------------------------

def test_correct_labels(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    visits.correct(a, "not_bird")
    assert store.get(b).is_bird is False and store.get(b).species is None and store.get(b).manual_is_bird is False
    visits.correct(b, "bird_unknown")
    assert store.get(a).is_bird is True and store.get(a).species is None and store.get(a).manual_is_bird is True
    visits.correct(a, "species:phoenicurus ochruros")
    assert store.get(b).species == "Black Redstart" and store.get(b).manual_species == "Black Redstart"
    visits.correct(a, "clear")
    assert store.get(a).manual_at is None and store.get(b).species == "Black Redstart"
    with pytest.raises(ValueError):
        visits.correct(a, "species:Dodo")
    with pytest.raises(ValueError):
        visits.correct(a, "nonsense")


def test_correct_non_bird_to_species_creates_visit_without_alert(store, visits):
    a = add(store, 0, bird=False); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:European Robin")
    c = store.get(a)
    assert c.visit_id is not None and c.is_bird is True and c.species == "European Robin"
    assert store.get_visit(c.visit_id).alert == "skipped" and visits._notifier.calls == []


def test_corrections_never_alert(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:European Robin")
    visits.tick(T0 + timedelta(seconds=61))
    assert visits._notifier.calls == [(a, "European Robin", True)]   # the close alert, with the corrected name, once


# --- deletes -----------------------------------------------------------------

def test_photo_removed_reresolves_or_drops_visit(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(1)); visits.photo_classified(store.get(b), live=True)
    vid = store.get(a).visit_id
    store.delete_capture(b); visits.photo_removed(vid)
    assert store.get_visit(vid) is not None
    store.delete_capture(a); visits.photo_removed(vid)
    assert store.get_visit(vid) is None and store.count_visits_with_species("Black Redstart") == 0


# --- without embeddings (migration window) ----------------------------------

def test_vote_falls_back_to_model_species_majority(store, visits):
    a = add(store, 0, species="Robin", p=0.7); visits.photo_classified(store.get(a), live=False)
    b = add(store, 10, species="Robin", p=0.6); visits.photo_classified(store.get(b), live=False)
    c = add(store, 20, species="Hoopoe", p=0.9); visits.photo_classified(store.get(c), live=False)
    assert store.get(c).species == "Robin" and store.get_visit(store.get(a).visit_id).species == "Robin"
```

(`live` fixture from conftest; it has `visits.gap_seconds = 60` and `species.threshold = 0.5` by default.)

- [x] **Step 2: Run** → `ModuleNotFoundError: birdy.visits`.

- [x] **Step 3: Implement** `birdy/visits.py`:

```python
"""Visits: one bird's photos within `visits.gap_seconds`, one label, one alert, one correction.

All decisions here are arithmetic on stored embeddings; no model runs. One RLock serialises
assign/merge/resolve/close/correct across the pipeline and web threads."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from birdy.live import LiveConfig
from birdy.species import Bank, score_candidates
from birdy.store import Capture, Store, Visit

log = logging.getLogger(__name__)

KEEP_OWN = 0.8            # a photo keeps its own species over the vote at or above this (second bird in frame)
ALERT_AFTER = 300         # seconds a visit may stay open before it alerts with the vote so far
LATE = timedelta(minutes=10)   # a visit whose last photo is older than this at decision time never alerts


@dataclass(frozen=True)
class Vote:
    species: str
    p: float
    pooled: np.ndarray


@dataclass(frozen=True)
class Change:
    capture_id: int
    old_species: str | None
    new_species: str | None
    captured_at: datetime


def vote(embeddings: list[np.ndarray], bank: Bank, cand_rows: np.ndarray, threshold: float) -> Vote | None:
    """Pooled (mean, renormalised) embedding scored over the candidates; None below `threshold`."""
    if not embeddings:
        return None
    pooled = np.mean(np.stack(embeddings), axis=0)
    n = float(np.linalg.norm(pooled))
    if n == 0.0:
        return None
    pooled = pooled / n
    scored = score_candidates(bank, cand_rows, pooled)
    if not scored or scored[0][1] < threshold:
        return None
    row, p = scored[0]
    return Vote(bank.common[row] or bank.latin[row], p, pooled)


def final_label(*, manual_is_bird: bool | None, manual_species: str | None, own: tuple[str, float] | None,
                vote_species: str | None) -> tuple[bool, str | None]:
    """(is_bird, species) for one photo. Priority: correction > keep-own > vote > unnamed."""
    if manual_is_bird is False:
        return False, None
    if manual_is_bird is True:
        return True, manual_species
    if own is not None and vote_species is not None and own[0] != vote_species and own[1] >= KEEP_OWN:
        return True, own[0]
    return True, vote_species


class Visits:
    def __init__(self, store: Store, live: LiveConfig, bank: Bank | None, species_list, notifier=None, publisher=None) -> None:
        self._store, self._live, self._bank, self._list = store, live, bank, species_list
        self._notifier, self._publisher = notifier, publisher
        self._lock = threading.RLock()

    # --- candidates ---------------------------------------------------------
    def _cand_rows(self) -> np.ndarray:
        if self._bank is None:
            return np.array([], np.int64)
        ids = self._list.candidates(self._live.get().species) if self._list is not None else None
        if ids is None:
            return np.arange(len(self._bank.ids))
        return np.array([i for i, t in enumerate(self._bank.ids) if int(t) in ids], np.int64)

    def _name_of_row(self, row: int) -> str:
        return self._bank.common[row] or self._bank.latin[row]

    def _own(self, emb: np.ndarray | None, cand: np.ndarray) -> list[tuple[int, float]]:
        return [] if emb is None or self._bank is None else score_candidates(self._bank, cand, emb)

    # --- entry points ----------------------------------------------------------
    def photo_classified(self, c: Capture, live: bool) -> list[Change]:
        """After `store_verdict`: a bird joins/creates a visit and the visit is resolved; a non-bird
        is resolved alone (final = model, no species)."""
        with self._lock:
            if not self._is_member(c):
                if c.visit_id is None:
                    return self.resolve_single(c.id)
                return self.resolve(c.visit_id)
            vid = c.visit_id if c.visit_id is not None else self.assign(c, live=live)
            return self.resolve(vid)

    def _is_member(self, c: Capture) -> bool:
        if c.manual_is_bird is not None:
            return bool(c.manual_is_bird)
        return bool(c.model_is_bird) or bool(c.is_bird)

    def assign(self, c: Capture, live: bool) -> int:
        gap = timedelta(seconds=self._live.get().visits.gap_seconds)
        found = [v for v in self._store.visits_overlapping(c.captured_at - gap, c.captured_at + gap)
                 if self._store.bird_members(v.id)]
        if not found:
            vid = self._store.create_visit(c.captured_at, c.captured_at, alert="pending" if live else "skipped")
        else:
            vid = found[0].id
            if len(found) > 1:
                self._store.merge_visits(vid, [v.id for v in found[1:]])
                states = {v.alert for v in found}
                self._store.set_visit_alert(vid, "sent" if "sent" in states else "skipped" if "skipped" in states else "pending")
        self._store.attach_to_visit(c.id, vid, extend=True)
        # inherit the visit's correction, if any
        corrected = next((p for p in self._store.visit_photos(vid) if p.manual_at is not None and p.id != c.id), None)
        if corrected is not None:
            self._store.set_manual(c.id, is_bird=bool(corrected.manual_is_bird), species=corrected.manual_species, at=corrected.manual_at)
        return vid

    def resolve_single(self, capture_id: int) -> list[Change]:
        c = self._store.get(capture_id)
        if c is None:
            return []
        is_bird = bool(c.manual_is_bird) if c.manual_is_bird is not None else bool(c.model_is_bird)
        species = (c.manual_species if c.manual_is_bird else None) if c.manual_is_bird is not None else None
        self._store.write_final(c.id, is_bird, c.model_confidence or 0.0, species, None)
        return []

    def resolve(self, visit_id: int) -> list[Change]:
        with self._lock:
            photos = self._store.visit_photos(visit_id)
            if not photos:
                self._store.delete_visit(visit_id)
                return []
            cfg = self._live.get()
            cand = self._cand_rows()
            members = [p for p in photos if self._is_member(p)]
            embs = self._store.embeddings([p.id for p in members])
            v = vote([embs[p.id] for p in members if p.id in embs], self._bank, cand, cfg.species.threshold) if self._bank is not None else None
            if v is None and members and not embs:
                v = self._majority(members)
            vote_species, vote_p = (v.species, v.p) if v is not None else (None, None)
            changes: list[Change] = []
            for p in photos:
                own_scored = self._own(embs.get(p.id), cand)
                own = (self._name_of_row(own_scored[0][0]), own_scored[0][1]) if own_scored else (
                    (p.model_species, p.model_species_confidence or 0.0) if p.model_species else None)
                is_bird, species = final_label(manual_is_bird=p.manual_is_bird, manual_species=p.manual_species, own=own, vote_species=vote_species)
                if not is_bird:
                    conf = p.model_confidence or 0.0; sp_conf = None
                elif p.manual_is_bird is True and species is not None:
                    conf = p.model_confidence or 0.0; sp_conf = conf
                else:
                    conf = p.model_confidence or 0.0
                    sp_conf = self._p_for(own_scored, species, p) if species else None
                if (p.is_bird, p.species) != (is_bird, species) or p.species_confidence != sp_conf or p.confidence != conf:
                    if p.species != species:
                        changes.append(Change(p.id, p.species, species, p.captured_at))
                        if p.published == "auto":
                            self._store.set_published(p.id, None)
                    self._store.write_final(p.id, is_bird, conf, species, sp_conf)
            self._store.set_visit_result(visit_id, vote_species, vote_p)
            self._publish(changes)
            return changes

    def _majority(self, members: list[Capture]) -> Vote | None:
        """No embeddings yet (migration window): the most common model species, ties by confidence."""
        named = [p for p in members if p.model_species]
        if not named:
            return None
        counts = Counter(p.model_species for p in named)
        best = max(counts, key=lambda s: (counts[s], max(p.model_species_confidence or 0 for p in named if p.model_species == s)))
        return Vote(best, max(p.model_species_confidence or 0 for p in named if p.model_species == best), np.zeros(1))

    def _p_for(self, scored: list[tuple[int, float]], species: str, p: Capture) -> float | None:
        for row, prob in scored:
            if self._name_of_row(row) == species:
                return prob
        return p.model_species_confidence if p.model_species == species else None

    def _publish(self, changes: list[Change]) -> None:
        if self._publisher is None:
            return
        for ch in changes:
            for sp in (ch.old_species, ch.new_species):
                if sp:
                    try:
                        self._publisher.evaluate_day(sp, ch.captured_at)
                    except Exception:
                        log.exception("publish re-rank failed for %s", sp)

    # --- close / alert -------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        gap = timedelta(seconds=self._live.get().visits.gap_seconds)
        with self._lock:
            for v in self._store.open_visits():
                if v.ended_at < now - gap:
                    self.resolve(v.id)
                    self._store.close_visit(v.id)
                    v = self._store.get_visit(v.id)
                    if v is not None and v.alert == "pending":
                        self._alert(v, now)
                elif v.alert == "pending" and now - v.started_at >= timedelta(seconds=ALERT_AFTER):
                    self.resolve(v.id)
                    self._alert(self._store.get_visit(v.id), now)

    def _alert(self, v: Visit, now: datetime) -> None:
        if now - v.ended_at > LATE or not v.species:
            self._store.set_visit_alert(v.id, "skipped")
            log.info("visit %d: alert skipped (%s)", v.id, "late" if v.species else "no species")
            return
        best = self._best_photo(v)
        if best is None or self._notifier is None:
            self._store.set_visit_alert(v.id, "skipped")
            return
        first_seen = self._store.count_visits_with_species(v.species) == 1
        try:
            reason = self._notifier.consider(best.id, self._store.photos_dir / best.path, v.species,
                                             best.species_confidence or v.species_confidence or 0.0, best.captured_at, first_seen)
        except Exception:
            log.exception("alert decision failed for visit %d", v.id)
            reason = None
        self._store.set_visit_alert(v.id, "sent" if reason else "skipped")
        if reason:
            log.info("alert queued for visit %d (%s, %s)", v.id, v.species, reason)

    def _best_photo(self, v: Visit) -> Capture | None:
        photos = [p for p in self._store.visit_photos(v.id) if p.is_bird and p.species == v.species]
        return max(photos, key=lambda p: (p.species_confidence or 0.0, -p.id), default=None)

    # --- corrections -----------------------------------------------------------------
    def correct(self, capture_id: int, label: str) -> None:
        """`not_bird` | `bird_unknown` | `species:<name>` | `clear`; ValueError for anything else."""
        with self._lock:
            c = self._store.get(capture_id)
            if c is None:
                raise KeyError(capture_id)
            if label == "clear":
                if c.visit_id is None:
                    self._store.clear_manual_single(c.id); self.resolve_single(c.id)
                else:
                    self._store.clear_manual(c.visit_id); self.resolve(c.visit_id)
                return
            if label == "not_bird":
                is_bird, species = False, None
            elif label == "bird_unknown":
                is_bird, species = True, None
            elif label.startswith("species:"):
                name = label[len("species:"):].strip()
                tid = self._bank.resolve(name) if self._bank is not None else None
                if tid is None:
                    raise ValueError(f"unknown species {name!r}")
                row = int(np.nonzero(self._bank.ids == tid)[0][0])
                is_bird, species = True, self._name_of_row(row)
            else:
                raise ValueError(f"bad label {label!r}")
            now = datetime.now(timezone.utc)
            if is_bird and c.visit_id is None:
                self._store.set_manual(c.id, is_bird=True, species=species, at=now)
                vid = self.assign(self._store.get(c.id), live=False)
            else:
                vid = c.visit_id
            targets = [p.id for p in self._store.visit_photos(vid)] if vid is not None else [c.id]
            for pid in targets:
                self._store.set_manual(pid, is_bird=is_bird, species=species, at=now)
            if vid is not None:
                self.resolve(vid)
            else:
                self.resolve_single(c.id)
            log.info("capture %d: corrected to %s (%d photos)", capture_id, label, len(targets))

    # --- deletes ------------------------------------------------------------------------
    def photo_removed(self, visit_id: int | None) -> None:
        if visit_id is None:
            return
        with self._lock:
            self.resolve(visit_id)   # deletes the visit when empty
```

Notes for the implementer: `Store.visits_overlapping` must return visits ordered by id so `found[0]` is the lowest id; `set_manual` must accept `at` so an inherited correction keeps the original time; `score_candidates` returns bank rows, so `_name_of_row` maps rows to names.

- [x] **Step 4: Run** `.venv/bin/pytest tests/test_visits.py -q` → pass. **Step 5: Commit** "Visits: grouping, embedding vote, resolve, tick/alerts, corrections".

---

### Task 5: Notifier `first_seen` parameter

**Files:** `birdy/notify.py`; `tests/test_notify.py`, `tests/test_pipeline.py` (fake signature)

- [x] **Step 1: Failing test** — in `tests/test_notify.py` change `test_first_seen_uses_store_count` into `test_consider_takes_first_seen_from_caller`: `consider(cid, path, "Robin", 0.9, when, first_seen=True)` → reason "new"; with `first_seen=False` and no watchlist → None. Update the other `consider` calls to the new positional/keyword shape `(capture_id, photo_path, species, species_confidence, captured_at, first_seen)`.
- [x] **Step 2: Run** → TypeError.
- [x] **Step 3: Implement** — signature `consider(self, capture_id, photo_path, species: str | None, species_confidence: float, captured_at, first_seen: bool, now=None)`; returns None when `not species` or alerts disabled; no store count. Update the docstring (fed by `Visits`, once per visit).
- [x] **Step 4: Run** `.venv/bin/pytest tests/test_notify.py -q` → pass. (Pipeline tests break until Task 7; that is expected — note it.) **Step 5: Commit** "Notify: consider() takes first_seen from the visit".

---

### Task 6: Publisher `evaluate_day`

**Files:** `birdy/publish.py`; `tests/test_publish.py`

- [x] **Step 1: Failing test** — append: `evaluate_day("Robin", when)` re-ranks that species/day (same effect as `evaluate(capture)` today; returns bool and notifies when changed); `evaluate(capture)` still works.
- [x] **Step 3: Implement** — `def evaluate_day(self, species: str, when: datetime) -> bool: with self._lock: changed = self._rank(species, when); if changed: self.notify(); return changed`; `evaluate(capture)` = guard + `return self.evaluate_day(capture.species, capture.captured_at)`.
- [x] **Step 4/5:** run, commit "Publish: evaluate_day".

---

### Task 7: Pipeline — verdicts through `Visits`, tick, no per-photo alerts

**Files:** `birdy/pipeline.py`; `tests/test_pipeline.py`

- [x] **Step 1: Failing tests** — in `tests/test_pipeline.py`:
  - `Pipeline(...)` gains `visits: Visits | None = None`; when None it builds `Visits(store, live, bank=None, species_list=None, notifier=notifier, publisher=publisher)` (model-species majority). Existing tests keep constructing `Pipeline(store, clf, live, notifier=n, publisher=pub)`.
  - `FakeNotifier.consider` takes `(capture_id, photo_path, species, species_confidence, captured_at, first_seen)` and records `(capture_id, photo_path, species, captured_at)`.
  - `FakePublisher` gains `evaluate_day(self, species, when)` appending `("day", species)` to `order` and `(species, when.date())` to a `days` list; keep `evaluate`/`forget`.
  - Replace `test_live_process_notifies` with `test_live_process_does_not_alert_until_the_visit_closes`: process a bird at T0 → `n.calls == []`; `p.visits.tick(T0 + 61 s)` → one call with that capture id. Keep `test_backfill_never_notifies` (sweep then tick → no calls because backfill visits are `skipped`).
  - `test_process_calls_publisher_with_stored_capture` → now asserts `pub.days == [("Robin", T0.date())]` (through `evaluate_day`).
  - The backfill tests: keep the never-demote, promotion, keep-species semantics, but the assertions on `forget`/`evaluate` become `days` assertions (old and new species' days re-ranked on a rename; `auto` cleared). Rewrite `test_backfill_rename_forgets_old_species_then_evaluates_new` → `..._reranks_old_and_new_days`.
  - New: `test_run_loop_ticks_visits` — with `sweep_interval=100`, start the pipeline, process a bird at `now − 2 min` (use a real timestamp), wait ≤ 8 s → the visit is closed (`store.open_visits() == []`).
- [x] **Step 3: Implement**
  - `process`: `store_verdict(...)` with `embedding=pred.embedding`; then `self.visits.photo_classified(self._store.get(capture_id), live=True)`; remove the notifier and publisher calls (Visits does both).
  - `_backfill`: the never-demote/known-bird logic stays, but the write becomes `store_verdict(...)` and then `changes = self.visits.photo_classified(self._store.get(c.id), live=False)`; `promoted` = `(not c.is_bird) and store.get(c.id).is_bird`; `renamed` = any change with old ≠ new; `got_species` = final species set. Remove the direct publisher/forget code (resolve handles it).
  - `_run`: add `next_tick`; every 5 s call `self.visits.tick()` inside try/except.
  - `Pipeline.visits` public attribute.
- [x] **Step 4:** `.venv/bin/pytest tests/test_pipeline.py -q` → pass; full suite. **Step 5: Commit** "Pipeline: verdicts flow through Visits; visits close and alert on a 5 s tick".

---

### Task 8: Web — corrections, Fix panel, visit filter, delete hooks

**Files:** `birdy/web/app.py`, `birdy/web/templates/index.html`, `birdy/web/static/osd.css`; `tests/test_web.py`

- [x] **Step 1: Failing tests** — `create_app(..., visits=None)`; the test app builds a real `Visits(store, live, bank=<3-species bank fixture>, species_list=FakeSpeciesList, notifier=None, publisher=FakePublisher)` and passes it. Tests:
  - `POST /captures/{id}/label` with `label=species:European Robin` → 303, row species "European Robin", `manual_species` set, whole visit updated; `label=bird_unknown` → species None, `manual_is_bird` True; `label=species:Dodo` → 400; `label=clear` → manual cleared; `not_bird` and 404/409 as before; the `next` rule.
  - GET `/?filter=birds` tile markup: a `<details class="fix">` with buttons `Not a bird`, `Bird, species unknown`, an `<input list="species-names">` and `Set`; a corrected tile shows `you` in the stamp and `model: Robin`; a multi-photo visit shows `visit · 2` linking to `/?visit=<id>`; `GET /?visit=<id>` lists only that visit's photos (and 400 for a non-int).
  - `<datalist id="species-names">` contains the local list names (from `visits`' species_list `current()`), most observations first, plus species seen.
  - Delete routes call `visits.photo_removed(visit_id)` (spy via a wrapper).
- [x] **Step 3: Implement**
  - `app.py`: `visits` param; `capture_label` accepts the four labels, maps `KeyError`→404, `ValueError`→400 (message in detail), 409 unless done; calls `visits.correct` in the threadpool; logs. Delete routes: after deleting, `visits.photo_removed(c.visit_id)` for each deleted capture. Grid route: `visit: int | None = None` query → `store.list_captures(filter="all", visit=visit)` (add `visit_id = ?` to `_where`), `visit_sizes` for the page's ids passed as `sizes`, `species_names` for the datalist.
  - `index.html`: in the stamps, `{% if c.manual_at %}<span class="tag you">you</span>{% endif %}`; when `c.manual_at and c.model_species` show `model: …` (already); `visit · N` link when `sizes[c.visit_id] > 1`. Replace the old label form with the Fix panel:
    ```jinja
    {% if c.status == 'done' and (c.is_bird or c.manual_at) %}
    <details class="fix"><summary class="pb">Fix</summary>
      <form method="post" action="/captures/{{ c.id }}/label" class="fixf">
        <input type="hidden" name="next" value="{{ here }}">
        <button class="pb" name="label" value="not_bird">Not a bird</button>
        <button class="pb" name="label" value="bird_unknown">Bird, species unknown</button>
        {% if c.manual_at %}<button class="pb" name="label" value="clear">Back to model</button>{% endif %}
        <input name="species_name" list="species-names" placeholder="species…" aria-label="Species name">
        <button class="pb" name="label" value="species" formaction="/captures/{{ c.id }}/label">Set</button>
      </form>
    </details>
    {% endif %}
    ```
    The route turns `label=species` + `species_name` into `species:<name>` (400 when the name is empty). One `<datalist id="species-names">` at the end of the page.
  - `osd.css`: `.fix summary` styled like `.pb`; `.fixf` wraps; `.tag.you` in amber per `DESIGN.md`'s burned-in rule (same as the `bird` tag colour).
- [x] **Step 4:** `.venv/bin/pytest tests/test_web.py -q`; full suite. **Step 5: Commit** "Web: Fix panel, species corrections, visit filter and badges".

---

### Task 9: Migration + retention hook + CLI wiring

**Files:** `birdy/store.py` (`migrate_visits`), `birdy/retention.py`, `birdy/cli.py`; `tests/test_store.py`, `tests/test_retention.py`, `tests/test_cli.py`

- [x] **Step 1: Failing tests**
  - `test_migrate_visits_builds_history(tmp_path)`: an old-format DB (create via `Store`, insert 6 photos: 3 birds at T0, T0+30 s, T0+45 s with species "Black Phoebe" (final columns set via `update_classification`), a non-bird, a corrected not-bird (`manual_at` set with `model_*` filled), a bird 10 min later); `n = store.migrate_visits(gap_seconds=60)` → returns the number of visits built (2); `data/birdy.db.bak-visits` exists; `model_species == species` on uncorrected rows and unchanged on the corrected one; the three early birds share a visit (`closed`, `alert == "skipped"`), the later one its own; `species_checked == False` on uncorrected birds (embedding backfill queued), True on the non-bird and the corrected row; `store.get_meta("visits_version") == "1"`; a second call returns 0 and changes nothing.
  - `tests/test_retention.py`: `run_once(store, days, visits=spy)` calls `visits.photo_removed(vid)` for each deleted row's visit id (`purge_non_birds` returns the deleted rows' `visit_id`s: change its return to `list[int | None]` plus count, or add a `deleted_visit_ids` out-param — pick returning `(count, visit_ids)` and update callers/tests).
  - `tests/test_cli.py`: in `test_cmd_run_passes_live_everywhere` monkeypatch `Store.migrate_visits` to record its `gap_seconds` and return 0; assert it was called with `config.visits.gap_seconds` before `Pipeline.__init__` (order list); when `migrate_visits` returns > 0 and publishing is enabled, `publisher.evaluate_all` is called (use a `FakePublisher` via monkeypatching `birdy.cli.Publisher`).
- [x] **Step 3: Implement**
  - `Store.migrate_visits(gap_seconds) -> int`: if `get_meta("visits_version") == "1"` → 0. Else `self._conn.backup(sqlite3.connect(data_dir / "birdy.db.bak-visits"))`; `BEGIN IMMEDIATE`; re-check meta; `UPDATE captures SET model_is_bird = is_bird, model_confidence = confidence, model_species = species, model_species_confidence = species_confidence WHERE manual_at IS NULL AND status = 'done'`; walk birds (`is_bird = 1 OR manual_is_bird = 1`) in `captured_at` order, opening a new visit when the gap to the previous bird exceeds `gap_seconds`, else extending; attach; `closed = 1, alert = 'skipped'`; `UPDATE captures SET species_checked = 0 WHERE is_bird = 1 AND manual_at IS NULL AND status = 'done' AND embedding IS NULL`; `set_meta("visits_version", "1")`; commit; `self.migrated_visits = True`; return the visit count.
  - `retention.run_once(store, days, now=None, visits=None)`: `count, vids = store.purge_non_birds(...)`; for each non-None id call `visits.photo_removed`. `RetentionJob(store, live, visits=None)` passes it.
  - `cli.cmd_run`: after `_classifier` and the species-list join: `visits = Visits(store, live, bank=clf.species.bank, species_list=clf.species_list, notifier=notifier, publisher=publisher)` (guard with getattr for test doubles → `bank=None`); `n = store.migrate_visits(config.visits.gap_seconds)`; log; `Pipeline(..., visits=visits)`; `RetentionJob(store, live, visits=visits)`; `create_app(..., visits=visits)`; after `pipeline.start()`, if `getattr(store, "migrated_visits", False) and publisher is not None: publisher.evaluate_all()`.
- [x] **Step 4:** full suite. **Step 5: Commit** "Visits: one-time history migration; retention and CLI wiring".

---

### Task 10: Docs, full suite

**Files:** `CLAUDE.md`, the spec

- [x] CLAUDE.md: architecture diagram gets `visits (group · vote · correct · alert)` between pipeline and store; a `birdy/visits.py` bullet (grouping, vote in embedding space, resolve priority with `KEEP_OWN`, tick every 5 s, `ALERT_AFTER`, late rule, corrections copied across the visit, deletes); the pipeline bullet (`store_verdict` writes `model_*`, final columns by `Visits.resolve`; `update_classification` = verdict + lone-photo resolve, used by tests); the notify bullet (fed by `Visits` once per visit; `first_seen` = first visit); the manual-delete / Not-a-bird bullets rewritten around the Fix panel and `POST /captures/{id}/label` labels; `[visits] gap_seconds` live; migration note (`meta.visits_version`, `birdy.db.bak-visits`, the embedding backfill after migration); the grid `?visit=` filter.
- [x] Spec: in "Data model → Writers", replace the `update_classification` sentence with the deviation stated at the top of this plan.
- [x] `.venv/bin/pytest -q` → all green. Commit "Docs: visits and corrections in CLAUDE.md".
- [ ] Deploy (from the Mac, when the user says so): `deploy/deploy.sh`; then on the Pi `journalctl -u birdy --since "-3 min" | grep -E "visits|migrat|backfill|alert|rror"` → expect `visits: built N visits over history`, a `species model`-style line about embeddings queued, backfill lines with `renamed`, no tracebacks; `/?filter=birds` shows `visit · N` badges and the Fix panel.
