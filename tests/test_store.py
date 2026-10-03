from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.store import Store


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


def test_purge_keeps_pending_and_error_rows(store):
    old = T0 - timedelta(days=10)
    pending = add(store, old)
    errored = add(store, old + timedelta(minutes=1))
    store.mark_error(errored)

    assert store.purge_non_birds(older_than_days=7, now=T0) == 0
    assert store.get(pending).status == "pending"
    assert store.get(errored).status == "error"


def test_count_pending(store):
    a = add(store, T0)
    add(store, T0 + timedelta(minutes=1))
    add(store, T0 + timedelta(minutes=2))
    store.update_classification(a, True, 0.9, None, None)
    assert store.count_pending() == 2
