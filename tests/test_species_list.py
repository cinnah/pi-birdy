import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from birdy.species import load_bank
from birdy.species_list import LocalList, SpeciesList, fetch_species_counts, needs_refresh, parse_species_counts

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_keeps_species_with_three_observations():
    data = json.loads((FIXTURES / "species_counts.json").read_text())
    rows = parse_species_counts([data])
    assert [r["id"] for r in rows] == [13094, 12942]
    assert rows[0] == {"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 120}
    assert rows[1]["common"] == "Erithacus rubecula"   # no English name -> latin


class FakeHttp:
    def __init__(self, pages_by_radius):
        self.pages_by_radius, self.calls = pages_by_radius, []

    def __call__(self, lat, lon, radius_km, page):
        self.calls.append((lat, lon, radius_km, page))
        return self.pages_by_radius[radius_km][page - 1]


def many(n, start=1000):
    return {"results": [{"count": 5, "taxon": {"id": start + i, "name": f"Sp {i}", "rank": "species", "preferred_common_name": f"Bird {i}"}} for i in range(n)]}


def test_fetch_pages_until_short_page():
    http = FakeHttp({50: [many(500), many(20, 2000)]})
    rows, used = fetch_species_counts(38.8, -9.39, 50, http=http, sleep=lambda s: None)
    assert used == 50 and len(rows) == 520 and [c[3] for c in http.calls] == [1, 2]


def test_fetch_retries_once_at_four_times_radius_when_too_few():
    http = FakeHttp({50: [many(5)], 200: [many(30)]})
    rows, used = fetch_species_counts(38.8, -9.39, 50, http=http, sleep=lambda s: None)
    assert used == 200 and len(rows) == 30 and [c[2] for c in http.calls] == [50, 200]


def test_local_list_roundtrip_atomic(tmp_path):
    lst = LocalList(fetched_at=datetime(2026, 11, 6, tzinfo=timezone.utc), lat=38.8, lon=-9.39, radius_km=50,
                    species=[{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 120}])
    lst.save(tmp_path / "species" / "local.json")
    back = LocalList.load(tmp_path / "species" / "local.json")
    assert back == lst and not list((tmp_path / "species").glob("*.tmp*"))
    assert LocalList.load(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert LocalList.load(tmp_path / "bad.json") is None


@pytest.mark.parametrize("change,expected", [
    ({}, False),
    ({"lat": 39.0}, True),
    ({"radius_km": 80}, True),
    ({"age_days": 91}, True),
])
def test_needs_refresh(change, expected):
    now = datetime(2026, 11, 6, tzinfo=timezone.utc)
    lst = LocalList(fetched_at=now - timedelta(days=change.get("age_days", 1)), lat=38.8, lon=-9.39, radius_km=50, species=[])
    assert needs_refresh(lst, change.get("lat", 38.8), -9.39, change.get("radius_km", 50), now) is expected
    assert needs_refresh(None, 38.8, -9.39, 50, now) is True


def bank_file(tmp_path):
    p = tmp_path / "bank.npz"
    np.savez(p, ids=np.array([13094, 12942, 1], np.int32),
             latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Upupa epops"]),
             common=np.array(["Black Redstart", "European Robin", "Eurasian Hoopoe"]),
             emb=np.eye(3, 4, dtype=np.float16), logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return load_bank(p)


class Cfg:
    def __init__(self, include=(), exclude=()):
        self.include, self.exclude = include, exclude


def test_candidates_from_list_plus_include_minus_exclude(tmp_path, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    assert sl.candidates(Cfg()) is None                      # no list, nothing configured -> whole bank
    LocalList(fetched_at=datetime.now(timezone.utc), lat=38.8, lon=-9.39, radius_km=50,
              species=[{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 1}]).save(tmp_path / "data" / "species" / "local.json")
    sl.reload()
    assert sl.candidates(Cfg()) == frozenset({13094})
    assert sl.candidates(Cfg(include=("eurasian hoopoe",), exclude=("Black Redstart",))) == frozenset({1})
    assert sl.candidates(Cfg(include=("Black Redstart",), exclude=("black redstart",))) == frozenset()   # exclude wins
    with caplog.at_level("WARNING"):
        assert sl.candidates(Cfg(include=("Dodo",))) == frozenset({13094})
        sl.candidates(Cfg(include=("Dodo",)))
    assert caplog.text.count("Dodo") == 1                     # warned once
    assert sl.unknown_names(("Dodo", "European Robin")) == ["Dodo"]


def test_unknown_names_reported_only_wants_the_stored_name(tmp_path):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    assert sl.unknown_names(["Phoenicurus ochruros"]) == []
    assert sl.unknown_names(["Phoenicurus ochruros"], reported_only=True) == ["Phoenicurus ochruros"]
    assert sl.unknown_names(["Black Redstart", " black redstart "], reported_only=True) == []


def test_candidates_without_list_but_with_exclude_uses_whole_bank(tmp_path):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    assert sl.candidates(Cfg(exclude=("Eurasian Hoopoe",))) == frozenset({13094, 12942})


def test_refresh_fetches_saves_and_bumps_version(tmp_path, monkeypatch):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: (parse_species_counts([many(25)]), r))
    v0 = sl.version
    assert sl.refresh_if_needed(38.8, -9.39, 50) is True
    assert sl.version == v0 + 1 and sl.current().radius_km == 50 and (tmp_path / "data" / "species" / "local.json").exists()
    assert sl.refresh_if_needed(38.8, -9.39, 50) is False     # fresh -> no fetch
    assert sl.refresh_if_needed(38.8, -9.39, 50, force=True) is True
    assert sl.refresh_if_needed(None, None, 50) is False      # no location -> no list


def test_refresh_failure_keeps_old_file(tmp_path, monkeypatch, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    old = LocalList(fetched_at=datetime.now(timezone.utc) - timedelta(days=100), lat=38.8, lon=-9.39, radius_km=50, species=[])
    old.save(tmp_path / "data" / "species" / "local.json")
    sl.reload()

    def boom(*a, **kw):
        raise OSError("down")

    monkeypatch.setattr("birdy.species_list.fetch_species_counts", boom)
    with caplog.at_level("WARNING"):
        assert sl.refresh_if_needed(38.8, -9.39, 50) is False
    assert "species list" in caplog.text and sl.current() == old


def test_refresh_thread_runs_once_and_stops(tmp_path, monkeypatch):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    ran = threading.Event()
    monkeypatch.setattr(sl, "refresh_if_needed", lambda *a, **kw: ran.set() or True)
    sl.start_refresh(38.8, -9.39, 50)
    assert ran.wait(2)
    sl.stop()


def test_fetch_stops_paging_at_the_cap(caplog):
    class Endless:
        calls = 0

        def __call__(self, lat, lon, radius_km, page):
            self.calls += 1
            return many(500)

    http = Endless()
    with caplog.at_level("WARNING"):
        rows, _ = fetch_species_counts(38.8, -9.39, 50, http=http, sleep=lambda s: None)
    assert http.calls == 20 and len(rows) == 20 * 500 and "20 pages" in caplog.text


def test_load_treats_naive_timestamp_as_utc(tmp_path):
    p = tmp_path / "local.json"
    p.write_text(json.dumps({"fetched_at": "2026-11-06T10:00:00", "lat": 38.8, "lon": -9.39, "radius_km": 50, "species": []}))
    lst = LocalList.load(p)
    assert lst is not None and lst.fetched_at.tzinfo is not None
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    assert needs_refresh(lst, 38.8, -9.39, 50, now) is False


def test_refresh_with_too_few_species_keeps_current(tmp_path, monkeypatch, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: (parse_species_counts([many(3)]), r * 4))
    v0 = sl.version
    with caplog.at_level("WARNING"):
        assert sl.refresh_if_needed(38.8, -9.39, 50) is False
    assert not (tmp_path / "data" / "species" / "local.json").exists()
    assert sl.current() is None and sl.version == v0 and "only 3 species within 200 km" in caplog.text


def test_refresh_stores_the_radius_used_and_the_configured_one(tmp_path, monkeypatch):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: (parse_species_counts([many(25)]), r * 4))
    assert sl.refresh_if_needed(38.8, -9.39, 50) is True
    cur = LocalList.load(tmp_path / "data" / "species" / "local.json")
    assert cur.radius_km == 200 and cur.configured_radius_km == 50
    assert sl.refresh_if_needed(38.8, -9.39, 50) is False     # a retry at 4x doesn't refetch at every start
    assert needs_refresh(cur, 38.8, -9.39, 200, datetime.now(timezone.utc)) is True


def test_refresh_save_failure_keeps_current(tmp_path, monkeypatch, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    old = LocalList(fetched_at=datetime.now(timezone.utc) - timedelta(days=100), lat=38.8, lon=-9.39, radius_km=50, species=[])
    old.save(tmp_path / "data" / "species" / "local.json")
    sl.reload()
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: (parse_species_counts([many(25)]), r))

    def no_disk(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(LocalList, "save", no_disk)
    with caplog.at_level("WARNING"):
        assert sl.refresh_if_needed(38.8, -9.39, 50) is False
    assert "species list refresh failed" in caplog.text and sl.current() == old
