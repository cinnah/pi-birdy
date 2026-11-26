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

    deleted, visit_ids = store.purge_non_birds(older_than_days=7, now=T0)

    assert deleted == 1 and visit_ids == []
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

    assert store.purge_non_birds(older_than_days=7, now=T0) == (1, [])
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
    assert [c.id for c in s.classification_backlog()] == [1]
    assert s.get(1).species_checked is False
    s.close()
    Store(data).close()  # second open: migration is a no-op


def test_update_classification_marks_species_checked(store):
    cid = add(store, T0)
    assert store.get(cid).species_checked is False
    store.update_classification(cid, True, 0.9, None, None)
    assert store.get(cid).species_checked is True
    assert store.classification_backlog() == []


def test_classification_backlog_newest_first_and_limit(store):
    ids = [add_species(store, T0 + timedelta(minutes=i), None, None) for i in range(3)]
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    assert [c.id for c in store.classification_backlog()] == ids[::-1]
    assert [c.id for c in store.classification_backlog(limit=1)] == [ids[2]]


def test_mark_checked(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.9, None, None)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    store.mark_checked(cid)
    assert store.classification_backlog() == []
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


def test_list_and_count_by_visit(store):
    a = add_species(store, T0, "Robin", 0.9)
    b = add_species(store, T0 + timedelta(seconds=30), "Robin", 0.8)
    add_species(store, T0 + timedelta(minutes=5), "Robin", 0.9)
    vid = store.create_visit(T0, T0)
    store.attach_to_visit(a, vid, extend=True)
    store.attach_to_visit(b, vid, extend=True)
    assert [x.id for x in store.list_captures("all", visit=vid)] == [b, a]
    assert store.count("all", visit=vid) == 2
    assert store.count("all", visit=vid + 1) == 0


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


def test_count_classification_backlog(store):
    for i in range(2):
        add_species(store, T0 + timedelta(minutes=i), None, None)
    assert store.count_classification_backlog() == 0
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0")
    assert store.count_classification_backlog() == 2


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


@pytest.fixture
def lisbon(monkeypatch):
    import time
    try:
        with monkeypatch.context() as m:
            m.setenv("TZ", "Europe/Lisbon")
            time.tzset()
            yield
    finally:
        time.tzset()


def test_species_days_one_entry_per_species_per_local_day(store, lisbon):
    # Same UTC date, different Lisbon days (UTC+1): 23:30 on the 4th and 00:30 on the 5th.
    a = datetime(2026, 10, 4, 22, 30, tzinfo=timezone.utc)
    b = datetime(2026, 10, 4, 23, 30, tzinfo=timezone.utc)
    _bird(store, a, "Robin", 0.9)
    _bird(store, b, "Robin", 0.9)
    _bird(store, datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc), "Wren", 0.9)   # same local day twice
    _bird(store, datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc), "Wren", 0.8)
    got = sorted(store.species_days())
    assert got == [
        ("Robin", a),
        ("Robin", b),
        ("Wren", datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc)),
    ]


def test_species_days_excludes_non_birds_unlabelled_and_pending(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    _bird(store, t, "Cat", 0.9, is_bird=False)
    _bird(store, t, None, None)
    store.add_capture(t, store.photo_path_for(t)[1], 1)   # pending
    assert store.species_days() == []


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
        "days": {today.isoformat(): 2},
        "hours": [0] * 9 + [2] + [0] * 14,
    }
    assert s["species"][1]["hours"][9] == 2 and sum(s["species"][1]["hours"]) == 2    # all time, not only the window
    assert s["species"][1]["name"] == "Wren" and s["species"][1]["count"] == 2
    assert s["species"][1]["days"] == {yday9.date().isoformat(): 1}   # the 40-day-old one is outside the window


def test_publish_stats_count_visits_not_photos(store):
    now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc).astimezone()
    today = now.date()
    t9, t15 = _local(today, 9), _local(today, 15)
    # one long visit: three Robin photos and, in the same visit, a second bird named by its own photo
    v1 = store.create_visit(t9, t9 + timedelta(seconds=40), alert="skipped")
    for i, sp in enumerate(["Robin", "Robin", "Wren", "Robin"]):
        store.attach_to_visit(_bird(store, t9 + timedelta(seconds=10 * i), sp, 0.9), v1)
    # a second Robin visit in the afternoon, two photos, one of them unnamed
    v2 = store.create_visit(t15, t15 + timedelta(seconds=10), alert="skipped")
    store.attach_to_visit(_bird(store, t15, None, None), v2)
    store.attach_to_visit(_bird(store, t15 + timedelta(seconds=10), "Robin", 0.9), v2)
    s = store.publish_stats(now)
    assert s["total_birds"] == 2                                   # two visits, not six photos
    assert s["per_day"][-1]["birds"] == 2 and s["per_hour"][9] == 1 and s["per_hour"][15] == 1
    robin, wren = s["species"]
    assert robin["name"] == "Robin" and robin["count"] == 2 and robin["days"] == {today.isoformat(): 2}
    assert robin["hours"][9] == 1 and robin["hours"][15] == 1 and sum(robin["hours"]) == 2
    assert robin["last"] == (t15 + timedelta(seconds=10)).isoformat(timespec="seconds")
    assert wren["count"] == 1 and wren["hours"][9] == 1          # the second bird in frame: one visit of its own name


def test_publish_stats_empty(store):
    s = store.publish_stats(datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc))
    assert s["total_birds"] == 0 and s["species"] == [] and s["first_capture"] is None
    assert len(s["per_day"]) == 30 and sum(d["birds"] for d in s["per_day"]) == 0


def test_purge_keeps_rows_with_any_published_state(store):
    old = T0 - timedelta(days=10)
    forced = add(store, old)
    plain = add(store, old + timedelta(minutes=1))
    for cid in (forced, plain):
        store.update_classification(cid, False, 0.1, None, None)
    store.set_published(forced, "forced")

    assert store.purge_non_birds(older_than_days=7, now=T0) == (1, [])
    assert store.get(forced).published == "forced"
    assert store.get(plain) is None


def test_delete_capture_removes_row_and_files(store):
    cid = add(store, T0)
    c = store.get(cid)
    photo, thumb = store.photos_dir / c.path, store.thumbs_dir / c.thumb_path
    assert photo.exists() and thumb.exists()
    deleted = store.delete_capture(cid)
    assert deleted is not None and deleted.id == cid
    assert store.get(cid) is None
    assert not photo.exists() and not thumb.exists()
    assert store.delete_capture(cid) is None  # already gone


def test_delete_non_birds_skips_birds_and_pending(store):
    bird, junk, err, pending = (add(store, T0 + timedelta(minutes=i)) for i in range(4))
    store.update_classification(bird, True, 0.9, "Robin", 0.8)
    store.update_classification(junk, False, 0.1, None, None)
    store.mark_error(err)
    gone = store.delete_non_birds([bird, junk, err, pending, 999])
    assert sorted(c.id for c in gone) == sorted([junk, err])
    assert store.get(bird) is not None and store.get(pending) is not None
    assert store.get(junk) is None and store.get(err) is None


def test_add_capture_stores_motion_box(store):
    _, rel = store.photo_path_for(T0)
    cid = store.add_capture(T0, rel, 100, motion_box=(0.25, 0.5, 0.125, 0.0625))
    assert store.get(cid).motion_box == (0.25, 0.5, 0.125, 0.0625)


def test_add_capture_without_motion_box(store):
    _, rel = store.photo_path_for(T0)
    cid = store.add_capture(T0, rel, 100)
    assert store.get(cid).motion_box is None


def test_old_database_gets_motion_columns(tmp_path):
    data = tmp_path / "old"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.execute(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending')"
    )
    conn.execute("INSERT INTO captures (captured_at, path, motion_score) VALUES ('2026-10-01T10:00:00.000+00:00', 'a.jpg', 5)")
    conn.commit()
    conn.close()
    s = Store(data)
    assert s.get(1).motion_box is None


def _classified_bird(store, when, species="Robin"):
    cid = add(store, when)
    store.update_classification(cid, True, 0.9, species, 0.8)
    return cid


def test_mark_not_bird_keeps_the_model_verdict(store):
    cid = _classified_bird(store, T0)
    store.set_published(cid, "auto")
    before = store.mark_not_bird(cid)
    assert before.is_bird and before.species == "Robin" and before.published == "auto"
    c = store.get(cid)
    assert c.is_bird is False and c.species is None and c.species_confidence is None and c.published is None
    assert c.manual_is_bird is False and c.manual_at is not None
    assert c.model_is_bird is True and c.model_species == "Robin" and c.model_species_confidence == 0.8
    assert store.count("birds", species="Robin") == 0


def test_clear_correction_restores_the_model_verdict(store):
    cid = _classified_bird(store, T0)
    store.mark_not_bird(cid)
    store.mark_not_bird(cid)  # twice: model_* still holds the model's verdict, not the correction
    c = store.clear_correction(cid)
    assert c.is_bird is True and c.species == "Robin" and c.species_confidence == 0.8 and c.confidence == 0.9
    assert c.manual_is_bird is None and c.manual_at is None


def test_correction_unknown_capture(store):
    assert store.mark_not_bird(999) is None
    assert store.clear_correction(999) is None


def test_corrected_photos_are_kept_by_retention_and_skipped_by_backfill(store):
    old = T0 - timedelta(days=30)
    cid = _classified_bird(store, old)
    store.mark_not_bird(cid)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0 WHERE id = ?", (cid,))
    assert store.purge_non_birds(older_than_days=7, now=T0) == (0, [])
    assert store.get(cid) is not None
    assert store.classification_backlog() == [] and store.count_classification_backlog() == 0


def test_update_classification_stores_detector_fields(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.6, None, None, det_confidence=0.6, det_box=(0.1, 0.2, 0.3, 0.4))
    c = store.get(cid)
    assert c.det_confidence == 0.6 and c.det_box == (0.1, 0.2, 0.3, 0.4)
    assert c.species_checked is True and store.classification_backlog() == []
    store.update_classification(cid, False, 0.0, None, None)
    c = store.get(cid)
    assert c.det_confidence == 0.0 and c.det_box is None


def test_backlog_includes_rows_not_yet_through_the_detector(store):
    cid = add(store, T0)
    store.update_classification(cid, False, 0.1, None, None)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET detector_checked = 0")
    assert [c.id for c in store.classification_backlog()] == [cid]
    assert store.count_classification_backlog() == 1
    store.mark_checked(cid)
    assert store.classification_backlog() == []


def test_migration_adds_detector_columns_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending', species_checked INTEGER NOT NULL DEFAULT 1);"
        "INSERT INTO captures (captured_at, path, motion_score, is_bird, status) VALUES ('2026-10-03T09:00:00.000+00:00', 'a.jpg', 1, 0, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    c = s.get(1)
    assert c.det_confidence is None and c.det_box is None
    assert [x.id for x in s.classification_backlog()] == [1]   # detector_checked defaults to 0
    s.close()
    Store(data).close()


def test_meta_roundtrip(store):
    assert store.get_meta("species_model") is None
    store.set_meta("species_model", "bioclip-2/int8/abc")
    assert store.get_meta("species_model") == "bioclip-2/int8/abc"
    store.set_meta("species_model", "x")
    assert store.get_meta("species_model") == "x"


def test_reset_species_for_model_marks_uncorrected_done_birds(store):
    bird = add_species(store, T0, "Black Phoebe", 0.8)
    nonbird = add(store, T0 + timedelta(minutes=1))
    store.update_classification(nonbird, False, 0.1, None, None)
    corrected = add_species(store, T0 + timedelta(minutes=2), "Robin", 0.9)
    store.mark_not_bird(corrected)
    pending = add(store, T0 + timedelta(minutes=3))
    assert store.reset_species_for_model("bioclip-2/int8/abc") == 1
    assert store.get(bird).species_checked is False and store.get(bird).species == "Black Phoebe"
    assert store.get(nonbird).species_checked is True and store.get(corrected).species_checked is True
    assert store.get(pending).status == "pending"
    assert store.get_meta("species_model") == "bioclip-2/int8/abc"
    assert store.reset_species_for_model("bioclip-2/int8/abc") == 0        # same model: idempotent
    store.update_classification(bird, True, 0.8, "Black Redstart", 0.9)
    assert store.reset_species_for_model("bioclip-2/int8/def") == 1        # new model: again
    assert store.get_meta("species_model") == "bioclip-2/int8/def"


def test_reset_species_counts_birds_in_the_backlog(store):
    bird = add_species(store, T0, "Black Phoebe", 0.8)
    store.reset_species_for_model("m1")
    assert [c.id for c in store.classification_backlog()] == [bird]
    assert store.count_classification_backlog() == 1


def test_store_verdict_writes_model_columns_only(store):
    cid = add(store, T0)
    emb = np.full(768, 0.036, np.float32)
    store.store_verdict(cid, is_bird=True, confidence=0.7, species="Robin", species_confidence=0.9,
                        det_confidence=0.6, det_box=(0.1, 0.2, 0.3, 0.4), embedding=emb)
    c = store.get(cid)
    assert c.status == "done" and c.species_checked and c.model_is_bird is True and c.model_confidence == 0.7
    assert c.model_species == "Robin" and c.model_species_confidence == 0.9 and c.det_confidence == 0.6
    assert c.is_bird is None and c.species is None            # final columns untouched
    assert c.has_embedding and np.allclose(store.embeddings([cid])[cid], emb, atol=1e-3)
    store.store_verdict(cid, is_bird=False, confidence=0.1, species=None, species_confidence=None, embedding=None)
    assert cid in store.embeddings([cid])                      # a None embedding keeps the stored one


def test_embeddings_returns_only_rows_with_a_blob(store):
    a = add(store, T0); b = add(store, T0 + timedelta(seconds=1)); c = add(store, T0 + timedelta(seconds=2))
    store.store_verdict(a, is_bird=True, confidence=0.7, species=None, species_confidence=None,
                        embedding=np.full(768, 0.5, np.float32))
    store.store_verdict(b, is_bird=True, confidence=0.7, species=None, species_confidence=None)
    out = store.embeddings([a, b, c, 999])
    assert set(out) == {a} and out[a].dtype == np.float32 and out[a].shape == (768,)
    assert store.embeddings([]) == {}


def test_update_classification_also_writes_model_columns(store):
    cid = add(store, T0)
    store.update_classification(cid, True, 0.9, "Robin", 0.8)
    c = store.get(cid)
    assert (c.model_is_bird, c.model_confidence, c.model_species, c.model_species_confidence) == (True, 0.9, "Robin", 0.8)


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
    assert store.count_visits_with_species("Robin") == 0          # no bird member yet
    store.store_verdict(a, is_bird=True, confidence=0.8, species="Robin", species_confidence=0.9)
    assert store.count_visits_with_species("Robin") == 1
    store.delete_visit(v1)
    assert store.get_visit(v1) is None and store.get(a).visit_id is None


def test_merge_into_unknown_visit_raises(store):
    a = add(store, T0)
    v = store.create_visit(T0, T0); store.attach_to_visit(a, v)
    with pytest.raises(KeyError):
        store.merge_visits(v + 1, [v])
    assert store.get_visit(v) is not None and store.get(a).visit_id == v


def test_merge_sets_alert_in_the_same_call(store):
    a = add(store, T0); b = add(store, T0 + timedelta(seconds=30))
    v1 = store.create_visit(T0, T0, alert="pending"); v2 = store.create_visit(T0 + timedelta(seconds=30), T0 + timedelta(seconds=30), alert="sent")
    store.attach_to_visit(a, v1); store.attach_to_visit(b, v2)
    with pytest.raises(ValueError):
        store.merge_visits(v1, [v2], alert="bogus")
    assert store.get_visit(v2) is not None
    store.merge_visits(v1, [v2], alert="sent")
    assert store.get_visit(v2) is None and store.get_visit(v1).alert == "sent" and store.get(b).visit_id == v1


def test_set_manual_visit_and_detach(store):
    a = add(store, T0); b = add(store, T0 + timedelta(seconds=5)); c = add(store, T0 + timedelta(minutes=5))
    v = store.create_visit(T0, T0 + timedelta(seconds=5)); store.attach_to_visit(a, v); store.attach_to_visit(b, v)
    at = T0 + timedelta(hours=1)
    store.set_manual_visit(v, is_bird=True, species="Robin", at=at)
    assert all(store.get(x).manual_species == "Robin" and store.get(x).manual_at == at for x in (a, b))
    assert store.get(c).manual_at is None
    store.detach_from_visit(b)
    assert store.get(b).visit_id is None and [x.id for x in store.visit_photos(v)] == [a]


def test_bird_members_ignore_the_final_column(store):
    a = add(store, T0)
    v = store.create_visit(T0, T0); store.attach_to_visit(a, v)
    store.store_verdict(a, is_bird=False, confidence=0.1, species=None, species_confidence=None)
    store.write_final(a, True, 0.9, "Robin", 0.9)        # a final bird the model calls a non-bird
    assert store.bird_members(v) == []


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


def test_visit_sizes_and_clear_manual_single(store):
    a = add(store, T0); b = add(store, T0 + timedelta(seconds=5)); c = add(store, T0 + timedelta(minutes=5))
    v1 = store.create_visit(T0 + timedelta(seconds=5), T0 + timedelta(seconds=5))
    v2 = store.create_visit(T0, T0, alert="sent")
    v3 = store.create_visit(T0 - timedelta(minutes=1), T0)
    store.attach_to_visit(a, v1, extend=True); store.attach_to_visit(b, v1); store.attach_to_visit(c, v2)
    assert store.get_visit(v1).started_at == T0                 # extend widens backwards too
    assert store.visit_sizes([v1, v2, v3]) == {v1: 2, v2: 1} and store.visit_sizes([]) == {}
    store.set_manual(a, is_bird=False, species=None); store.set_manual(b, is_bird=False, species=None)
    store.clear_manual_single(a)
    assert store.get(a).manual_at is None and store.get(b).manual_at is not None
    assert store.get_visit(999) is None


def test_unresolved_done_lists_classified_rows_without_final_columns(store):
    pending = add(store, T0)
    labelled = add(store, T0 + timedelta(minutes=1))
    store.update_classification(labelled, True, 0.9, "Robin", 0.8)
    a = add(store, T0 + timedelta(minutes=2))
    b = add(store, T0 + timedelta(minutes=3))
    for cid in (b, a):
        store.store_verdict(cid, is_bird=True, confidence=0.7, species=None, species_confidence=None)
    assert [c.id for c in store.unresolved_done()] == [a, b]
    assert [c.id for c in store.unresolved_done(limit=1)] == [a]
    assert pending not in [c.id for c in store.unresolved_done()]


def test_migrate_visits_builds_history(tmp_path):
    data = tmp_path / "data"
    store = Store(data)
    b1 = add_species(store, T0, "Black Phoebe", 0.8)
    b2 = add_species(store, T0 + timedelta(seconds=30), "Black Phoebe", 0.7)
    b3 = add_species(store, T0 + timedelta(seconds=45), "Black Phoebe", 0.9)
    nonbird = add_species(store, T0 + timedelta(seconds=50), None, None, is_bird=False)
    corrected = add_species(store, T0 + timedelta(seconds=55), "Robin", 0.6)
    store.mark_not_bird(corrected)
    # 100 s after the start but 55 s after b3: the gap chains from the previous bird, not the visit start.
    b4 = add_species(store, T0 + timedelta(seconds=100), "Robin", 0.95)
    late = add_species(store, T0 + timedelta(minutes=10), "House Sparrow", 0.5)
    # An old-format DB: uncorrected rows have no model_* yet (only a correction filled them).
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE captures SET model_is_bird = NULL, model_confidence = NULL, model_species = NULL,"
            " model_species_confidence = NULL WHERE manual_at IS NULL"
        )
    assert store.get(b1).model_species is None and store.migrated_visits is False

    assert store.migrate_visits(gap_seconds=60) == 2
    assert store.migrated_visits is True
    bak = data / "birdy.db.bak-visits"
    assert bak.exists()
    conn = sqlite3.connect(bak)
    try:   # taken before any write
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 7
        assert conn.execute("SELECT model_species FROM captures WHERE id = ?", (b1,)).fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'visits_version'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0
    finally:
        conn.close()
    for cid in (b1, b2, b3, b4, late, nonbird):
        c = store.get(cid)
        assert (c.model_is_bird, c.model_confidence, c.model_species, c.model_species_confidence) == (
            c.is_bird, c.confidence, c.species, c.species_confidence)
    fixed = store.get(corrected)
    assert fixed.model_is_bird is True and fixed.model_species == "Robin" and fixed.model_species_confidence == 0.6
    assert fixed.is_bird is False and fixed.species is None

    early = {store.get(cid).visit_id for cid in (b1, b2, b3, b4)}
    assert len(early) == 1 and None not in early
    v = store.get_visit(early.pop())
    assert v.closed and v.alert == "skipped" and v.species == "Black Phoebe" and v.species_confidence == 0.9
    assert v.started_at == store.get(b1).captured_at and v.ended_at == store.get(b4).captured_at
    lone = store.get_visit(store.get(late).visit_id)
    assert lone.id != v.id and lone.closed and lone.alert == "skipped" and lone.species == "House Sparrow"
    assert store.get(nonbird).visit_id is None and store.get(corrected).visit_id is None
    assert [c.id for c in store.visit_photos(v.id)] == [b1, b2, b3, b4]

    # Uncorrected birds have no embedding: queued for the backfill; the rest are untouched.
    assert all(store.get(cid).species_checked is False for cid in (b1, b2, b3, b4, late))
    assert store.get(nonbird).species_checked is True and store.get(corrected).species_checked is True
    assert store.get_meta("visits_version") == "1"

    def snapshot():
        with store._lock:
            return (
                [tuple(r) for r in store._conn.execute("SELECT * FROM captures ORDER BY id")],
                [tuple(r) for r in store._conn.execute("SELECT * FROM visits ORDER BY id")],
            )

    before = snapshot()
    assert store.migrate_visits(gap_seconds=60) == 0
    assert snapshot() == before
    store.close()
    again = Store(data)
    assert again.migrate_visits(gap_seconds=60) == 0 and again.migrated_visits is False
    again.close()


def test_migrate_visits_majority_species_and_existing_visits(store):
    a = add_species(store, T0, "Robin", 0.4)
    b = add_species(store, T0 + timedelta(seconds=10), "Black Phoebe", 0.9)
    c = add_species(store, T0 + timedelta(seconds=20), "Robin", 0.5)
    d = add_species(store, T0 + timedelta(seconds=30), None, None)          # a bird with no species
    kept = add_species(store, T0 + timedelta(hours=1), "Wren", 0.7)
    vid = store.create_visit(T0 + timedelta(hours=1), T0 + timedelta(hours=1))
    store.attach_to_visit(kept, vid)
    assert store.migrate_visits(gap_seconds=60) == 1
    v = store.get_visit(store.get(a).visit_id)
    assert v.species == "Robin" and v.species_confidence == 0.5
    assert {store.get(x).visit_id for x in (a, b, c, d)} == {v.id}
    assert store.get(kept).visit_id == vid and store.get_visit(vid).alert == "pending"   # left alone


def test_migrate_visits_rolls_back_on_failure(store, monkeypatch):
    b = add_species(store, T0, "Black Phoebe", 0.8)
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET model_is_bird = NULL, model_confidence = NULL,"
                            " model_species = NULL, model_species_confidence = NULL")

    def boom(rows):
        raise RuntimeError("boom")

    monkeypatch.setattr("birdy.store._majority_species", boom)
    with pytest.raises(RuntimeError):
        store.migrate_visits(gap_seconds=60)
    assert store.get_meta("visits_version") is None and store.migrated_visits is False
    with store._lock:
        assert store._conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0
    c = store.get(b)
    assert c.model_is_bird is None and c.model_species is None and c.visit_id is None and c.species_checked is True
    monkeypatch.undo()
    assert store.migrate_visits(gap_seconds=60) == 1
    assert store.get(b).model_species == "Black Phoebe" and store.get_meta("visits_version") == "1"


def test_count_visits_with_species_ignores_visits_without_a_bird(store):
    a = add_species(store, T0, "Robin", 0.9)
    vid = store.create_visit(T0, T0)
    store.attach_to_visit(a, vid)
    store.set_visit_result(vid, "Robin", 0.9)
    assert store.count_visits_with_species("Robin") == 1
    store.delete_capture(a)                 # its only member is gone; the visit row lingers
    assert store.get_visit(vid) is not None and store.count_visits_with_species("Robin") == 0


def _eos_rel(store, cid) -> str:
    c = store.get(cid)
    rel = c.path.replace(".jpg", "_eos.jpg")
    write_jpeg(store.photos_dir / rel, w=1200, h=800)
    return rel


def _verdict(store, cid, **kw):
    store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None, **kw)


def test_photo_is_the_pi_photo_by_default(store):
    cid = add(store, T0)
    _verdict(store, cid)
    c = store.get(cid)
    assert c.dslr_path is None and c.dslr_box is None and c.photo == c.path


def test_promoted_capture_uses_the_dslr_photo_and_thumb(store):
    cid = add(store, T0)
    old_thumb = store.thumbs_dir / store.get(cid).thumb_path
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    c = store.get(cid)
    assert c.dslr_path == rel and c.dslr_box == (0.4, 0.4, 0.1, 0.1)
    assert c.photo == rel and c.path != rel
    assert c.thumb_path == rel and (store.thumbs_dir / rel).exists()
    assert not old_thumb.exists()


def test_dslr_frame_without_a_bird_is_not_the_photo(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=None)
    c = store.get(cid)
    assert c.dslr_path == rel and c.dslr_box is None and c.photo == c.path
    assert c.thumb_path == c.path


def test_a_later_verdict_keeps_the_dslr_columns(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    _verdict(store, cid)                      # the backfill: no dslr arguments
    assert store.get(cid).photo == rel


def test_count_dslr_shots(store):
    a, b = add(store, T0), add(store, T0 + timedelta(hours=1))
    _verdict(store, a, dslr_path=_eos_rel(store, a), dslr_box=None)
    _verdict(store, b, dslr_path=_eos_rel(store, b), dslr_box=(0.1, 0.1, 0.2, 0.2))
    assert store.count_dslr_shots(T0 - timedelta(minutes=1)) == 2
    assert store.count_dslr_shots(T0 + timedelta(minutes=30)) == 1


def test_delete_capture_removes_the_dslr_file(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    store.delete_capture(cid)
    assert not (store.photos_dir / rel).exists() and not (store.thumbs_dir / rel).exists()


def test_purge_dslr_unused(store):
    old, new, promoted = add(store, T0), add(store, T0 + timedelta(days=9)), add(store, T0 + timedelta(seconds=5))
    rels = {}
    for cid, box in ((old, None), (new, None), (promoted, (0.4, 0.4, 0.1, 0.1))):
        rels[cid] = _eos_rel(store, cid)
        _verdict(store, cid, dslr_path=rels[cid], dslr_box=box)
    n = store.purge_dslr_unused(older_than_days=7, now=T0 + timedelta(days=10))
    assert n == 1
    assert store.get(old).dslr_path is None and not (store.photos_dir / rels[old]).exists()
    assert store.get(new).dslr_path == rels[new]                       # too recent
    assert store.get(promoted).photo == rels[promoted]                 # a bird photo: kept
    assert (store.photos_dir / store.get(old).path).exists()           # the Pi photo is untouched


def test_species_summary_best_photo_is_the_main_photo(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    store.write_final(cid, True, 0.9, "Robin", 0.8)
    assert store.species_summary()[0].best_path == rel


def test_a_photo_taken_by_hand_uses_the_dslr_frame_without_a_box_and_is_not_purged_as_unused(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=None, dslr_main=True)
    c = store.get(cid)
    assert c.dslr_main is True and c.dslr_box is None and c.photo == rel and c.thumb_path == rel
    assert store.purge_dslr_unused(older_than_days=7, now=T0 + timedelta(days=30)) == 0
    assert (store.photos_dir / rel).exists()
    store.delete_capture(cid)
    assert not (store.photos_dir / rel).exists()
