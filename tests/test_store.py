import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.store import SpeciesRow, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def write_jpeg(path: Path, w=800, h=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 120, dtype=np.uint8))


def add(store: Store, when: datetime, score=200) -> int:
    abs_path, rel_path = store.photo_path_for(when)
    write_jpeg(abs_path)
    return store.add_capture(when, rel_path, score)


# Local-time aware, because photo paths use local time.
T0 = datetime(2026, 10, 3, 10, 0, 0, 123000).astimezone()


def test_creates_dirs_and_db(store, tmp_path):
    assert (tmp_path / "data" / "birdy.db").exists()
    assert store.photos_dir.is_dir() and store.thumbs_dir.is_dir()


def test_photo_path_layout(store):
    abs_path, rel_path = store.photo_path_for(T0)
    assert rel_path == Path("2026/10/03/100000_123.jpg")
    assert abs_path == store.photos_dir / rel_path
    assert abs_path.parent.is_dir()


def test_add_capture_creates_thumbnail_and_pending_row(store):
    cid = add(store, T0)
    c = store.get(cid)
    assert c.status == "pending" and c.is_bird is None
    assert c.motion_score == 200
    thumb = store.thumbs_dir / c.thumb_path
    assert thumb.exists()
    assert cv2.imread(str(thumb)).shape[1] == 400


def test_update_classification(store):
    cid = add(store, T0)
    store.update_classification(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None)
    c = store.get(cid)
    assert c.status == "done" and c.is_bird is True and c.confidence == 0.9


def test_mark_error(store):
    cid = add(store, T0)
    store.mark_error(cid)
    assert store.get(cid).status == "error"


def test_list_filters_and_orders_newest_first(store):
    a = add(store, T0)
    b = add(store, T0 + timedelta(minutes=1))
    c = add(store, T0 + timedelta(minutes=2))
    store.update_classification(a, True, 0.9, None, None)
    store.update_classification(b, False, 0.1, None, None)
    assert [x.id for x in store.list_captures("all")] == [c, b, a]
    assert [x.id for x in store.list_captures("birds")] == [a]
    assert [x.id for x in store.list_captures("non-birds")] == [b]
    assert store.count("all") == 3 and store.count("birds") == 1


def test_list_pagination(store):
    ids = [add(store, T0 + timedelta(minutes=i)) for i in range(5)]
    page = store.list_captures("all", limit=2, offset=2)
    assert [x.id for x in page] == [ids[2], ids[1]]


def test_pending_lists_only_pending(store):
    a = add(store, T0)
    b = add(store, T0 + timedelta(minutes=1))
    store.update_classification(a, True, 0.9, None, None)
    assert [x.id for x in store.pending()] == [b]


def test_purge_deletes_only_old_non_birds(store):
    old = T0 - timedelta(days=10)
    old_bird = add(store, old)
    old_junk = add(store, old + timedelta(minutes=1))
    new_junk = add(store, T0)
    store.update_classification(old_bird, True, 0.9, None, None)
    store.update_classification(old_junk, False, 0.1, None, None)
    store.update_classification(new_junk, False, 0.1, None, None)
    junk = store.get(old_junk)

    deleted = store.purge_non_birds(older_than_days=7, now=T0)

    assert deleted == 1
    assert store.get(old_junk) is None
    assert not (store.photos_dir / junk.path).exists()
    assert not (store.thumbs_dir / junk.thumb_path).exists()
    assert store.get(old_bird) is not None and store.get(new_junk) is not None


def test_site_roundtrip(store):
    store.set_site("home", 38.7, -9.1)
    assert store.site() == {"name": "home", "lat": 38.7, "lon": -9.1}
    store.set_site("home2", None, None)
    assert store.site() == {"name": "home2", "lat": None, "lon": None}


def test_last_capture_at(store):
    assert store.last_capture_at() is None
    add(store, T0)
    assert store.last_capture_at() == T0


def test_disk_free_is_positive(store):
    assert store.disk_free_bytes() > 0


def test_add_capture_with_unreadable_photo_still_inserts(store):
    abs_path, rel_path = store.photo_path_for(T0)
    abs_path.write_bytes(b"not a jpeg")
    cid = store.add_capture(T0, rel_path, 200)
    c = store.get(cid)
    assert c.status == "pending" and c.thumb_path is None


def test_unknown_filter_raises_key_error(store):
    with pytest.raises(KeyError):
        store.list_captures("cats")
    with pytest.raises(KeyError):
        store.count("cats")


def test_purge_keeps_pending_and_deletes_old_error_rows(store):
    old = T0 - timedelta(days=10)
    pending = add(store, old)
    errored = add(store, old + timedelta(minutes=1))
    store.mark_error(errored)
    recent_error = add(store, T0 - timedelta(days=1))
    store.mark_error(recent_error)

    assert store.purge_non_birds(older_than_days=7, now=T0) == 1
    assert store.get(pending).status == "pending"
    assert store.get(errored) is None
    assert store.get(recent_error).status == "error"


def test_count_pending(store):
    a = add(store, T0)
    add(store, T0 + timedelta(minutes=1))
    add(store, T0 + timedelta(minutes=2))
    store.update_classification(a, True, 0.9, None, None)
    assert store.count_pending() == 2



def add_species(store, when, species, conf, is_bird=True):
    cid = add(store, when)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return cid


def test_migration_adds_species_checked_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending');"
        "INSERT INTO captures (captured_at, path, motion_score, is_bird, status) VALUES ('2026-10-03T09:00:00.000+00:00', 'a.jpg', 1, 1, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    assert [c.id for c in s.species_backlog()] == [1]
    assert s.get(1).species_checked is False
    s.close()
    Store(data).close()  # second open: migration is a no-op


def test_update_classification_marks_species_checked(store):
    cid = add(store, T0)
    assert store.get(cid).species_checked is False
    store.update_classification(cid, True, 0.9, None, None)
    assert store.get(cid).species_checked is True
    assert store.species_backlog() == []


def test_species_backlog_newest_first_and_limit(store):
    ids = [add_species(store, T0 + timedelta(minutes=i), None, None) for i in range(3)]
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    assert [c.id for c in store.species_backlog()] == ids[::-1]
    assert [c.id for c in store.species_backlog(limit=1)] == [ids[2]]


def test_mark_species_checked(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.9, None, None)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    store.mark_species_checked(cid)
    assert store.species_backlog() == []
    assert store.get(cid).status == "done"


def test_species_summary(store):
    add_species(store, T0, "Robin", 0.6)
    b = add_species(store, T0 + timedelta(minutes=1), "Robin", 0.9)
    c = add_species(store, T0 + timedelta(minutes=2), "Blue Tit", 0.7)
    add_species(store, T0 + timedelta(minutes=3), None, None)
    rows = store.species_summary()
    assert [r.species for r in rows] == ["Blue Tit", "Robin"]  # last seen desc
    robin = rows[1]
    assert isinstance(robin, SpeciesRow)
    assert robin.count == 2 and robin.first_seen == T0 and robin.last_seen == T0 + timedelta(minutes=1)
    assert robin.best_capture_id == b and robin.best_thumb_path == store.get(b).thumb_path
    assert rows[0].best_capture_id == c


def test_species_summary_empty(store):
    assert store.species_summary() == []


def test_list_and_count_by_species(store):
    r1 = add_species(store, T0, "Robin", 0.9)
    add_species(store, T0 + timedelta(minutes=1), "Blue Tit", 0.9)
    assert [x.id for x in store.list_captures("all", species="Robin")] == [r1]
    assert store.count("all", species="Robin") == 1
    assert store.count("birds", species="Blue Tit") == 1
    assert store.count("non-birds", species="Blue Tit") == 0


def test_last_species(store):
    assert store.last_species() is None
    add_species(store, T0, "Robin", 0.9)
    add_species(store, T0 + timedelta(minutes=5), None, None)
    name, when = store.last_species()
    assert name == "Robin" and when == T0


def test_fresh_db_has_species_checked_column(store):
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(captures)")}
    assert "species_checked" in cols


def test_species_summary_null_confidence_loses(store):
    scored = add_species(store, T0, "Robin", 0.5)
    add_species(store, T0 + timedelta(minutes=1), "Robin", None)  # newer, but no confidence
    (robin,) = store.species_summary()
    assert robin.best_capture_id == scored


def test_species_summary_excludes_non_bird_rows(store):
    add_species(store, T0, "Robin", 0.9, is_bird=False)
    add_species(store, T0 + timedelta(minutes=1), "Blue Tit", 0.8)
    assert [r.species for r in store.species_summary()] == ["Blue Tit"]


def test_last_species_ignores_non_birds(store):
    add_species(store, T0, "Robin", 0.9)
    add_species(store, T0 + timedelta(minutes=5), "Blue Tit", 0.9, is_bird=False)
    assert store.last_species() == ("Robin", T0)


def test_species_filter_excludes_non_bird_rows(store):
    add_species(store, T0, "Robin", 0.9, is_bird=False)
    bird = add_species(store, T0 + timedelta(minutes=1), "Robin", 0.9)
    assert [c.id for c in store.list_captures("all", species="Robin")] == [bird]
    assert store.count("all", species="Robin") == 1


def test_count_species_backlog(store):
    for i in range(2):
        add_species(store, T0 + timedelta(minutes=i), None, None)
    assert store.count_species_backlog() == 0
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    assert store.count_species_backlog() == 2


def _bird(store, when, species, conf, is_bird=True):
    _, rel = store.photo_path_for(when)
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return cid


def test_fresh_db_has_publish_columns(store):
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(captures)")}
    assert {"published", "web_path"} <= cols
    assert "captures_published" in {r["name"] for r in store._conn.execute("PRAGMA index_list(captures)")}


def test_migration_adds_publish_columns_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending', species_checked INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO captures (captured_at, path, motion_score, status) VALUES ('2026-10-01T10:00:00.000+00:00', 'a.jpg', 1, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    c = s.get(1)
    assert c.published is None and c.web_path is None


def test_set_published_and_manual_states(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    a = _bird(store, t, "Robin", 0.9)
    b = _bird(store, t + timedelta(minutes=1), "Robin", 0.8)
    c = _bird(store, t + timedelta(minutes=2), "Robin", 0.7)
    store.set_published(a, "auto")
    store.set_published(b, "forced")
    store.set_published(c, "hidden")
    assert [x.id for x in store.published()] == [b, a]          # newest first, hidden excluded
    assert store.manual_states() == {b: "forced", c: "hidden"}
    store.set_published(a, None)
    assert store.get(a).published is None
    with pytest.raises(ValueError):
        store.set_published(a, "maybe")


def test_species_day_candidates_window_and_order(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    low = _bird(store, t, "Robin", 0.5)
    high = _bird(store, t + timedelta(minutes=1), "Robin", 0.9)
    _bird(store, t, "Wren", 0.9)                           # other species
    _bird(store, t + timedelta(days=1), "Robin", 0.95)     # outside the window
    nb = _bird(store, t + timedelta(minutes=2), "Robin", 0.99, is_bird=False)  # non-bird with a label
    pend = store.add_capture(t + timedelta(minutes=3), store.photo_path_for(t + timedelta(minutes=3))[1], 1)  # pending
    got = store.species_day_candidates("Robin", t, t + timedelta(days=1))
    assert [c.id for c in got] == [high, low]
    assert nb not in [c.id for c in got] and pend not in [c.id for c in got]


def test_set_web_path_and_stale_web_paths(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    a = _bird(store, t, "Robin", 0.9)
    b = _bird(store, t + timedelta(minutes=1), "Robin", 0.9)
    store.set_published(a, "auto")
    store.set_web_path(a, "2026/10/04/x.jpg")
    store.set_web_path(b, "2026/10/04/y.jpg")   # has a web copy but is not published
    assert store.get(a).web_path == "2026/10/04/x.jpg"
    assert store.stale_web_paths() == [(b, "2026/10/04/y.jpg")]
    store.set_published(a, "hidden")
    assert sorted(store.stale_web_paths()) == [(a, "2026/10/04/x.jpg"), (b, "2026/10/04/y.jpg")]


def _local(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m).astimezone()   # naive -> local zone, DST-correct


def test_publish_stats(store):
    now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc).astimezone()
    today = now.date()
    today9 = _local(today, 9)
    yday9 = _local(today - timedelta(days=1), 9)
    old9 = _local(today - timedelta(days=40), 9)
    _bird(store, today9, "Robin", 0.9)
    _bird(store, today9 + timedelta(minutes=5), "Robin", 0.8)
    _bird(store, yday9, "Wren", 0.9)
    _bird(store, old9, "Wren", 0.9)   # outside per_day but counted in totals
    _bird(store, today9, None, None)                          # bird without species
    _bird(store, today9, "Cat", 0.9, is_bird=False)           # not a bird: ignored everywhere
    s = store.publish_stats(now)
    assert s["total_birds"] == 5
    assert s["species_count"] == 2
    assert s["first_capture"] == old9.date().isoformat()
    assert s["last_capture_at"] == (today9 + timedelta(minutes=5)).isoformat(timespec="seconds")
    assert len(s["per_day"]) == 30 and s["per_day"][-1] == {"day": now.date().isoformat(), "birds": 3}
    assert s["per_day"][-2]["birds"] == 1
    assert len(s["per_hour"]) == 24 and s["per_hour"][9] == 5
    assert s["species"][0] == {
        "name": "Robin", "count": 2,
        "first": today9.date().isoformat(), "last": (today9 + timedelta(minutes=5)).isoformat(timespec="seconds"),
    }
    assert s["species"][1]["name"] == "Wren" and s["species"][1]["count"] == 2


def test_publish_stats_empty(store):
    s = store.publish_stats(datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc))
    assert s["total_birds"] == 0 and s["species"] == [] and s["first_capture"] is None
    assert len(s["per_day"]) == 30 and sum(d["birds"] for d in s["per_day"]) == 0
