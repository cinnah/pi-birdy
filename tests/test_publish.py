"""Publish policy and Publisher (mirror, manifest, rsync, thread)."""

from datetime import datetime, timedelta, timezone

from birdy.publish import PublishSettings, local_day_bounds, rank_day
from birdy.store import Capture

T0 = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
S = PublishSettings(min_confidence=0.6, per_species_per_day=2)


def cap(i, conf, published=None, species="Robin"):
    return Capture(
        id=i, captured_at=T0 + timedelta(minutes=i), path=f"p{i}.jpg", thumb_path=None, motion_score=1,
        is_bird=True, confidence=0.9, species=species, species_confidence=conf, status="done",
        species_checked=True, published=published,
    )


def test_rank_day_top_n_by_confidence_then_id():
    got = rank_day([cap(1, 0.7), cap(2, 0.9), cap(3, 0.9), cap(4, 0.8)], {}, S)
    assert got == {2: "auto", 3: "auto", 4: None, 1: None}


def test_rank_day_threshold():
    got = rank_day([cap(1, 0.59), cap(2, 0.6), cap(3, None)], {}, S)
    assert got == {1: None, 2: "auto", 3: None}


def test_rank_day_demotes_weaker_auto():
    got = rank_day([cap(1, 0.7, "auto"), cap(2, 0.8, "auto"), cap(3, 0.95)], {}, S)
    assert got == {3: "auto", 2: "auto", 1: None}


def test_rank_day_manual_states_win_and_forced_does_not_take_a_slot():
    manual = {1: "forced", 2: "hidden"}
    got = rank_day([cap(1, 0.1), cap(2, 0.99), cap(3, 0.7), cap(4, 0.65), cap(5, 0.61)], manual, S)
    assert got == {1: "forced", 2: "hidden", 3: "auto", 4: "auto", 5: None}


def test_rank_day_n_bounds():
    caps = [cap(i, 0.9) for i in range(1, 25)]
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 1)).values()) == 1
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 20)).values()) == 20


def test_rank_day_empty():
    assert rank_day([], {}, S) == {}


def test_local_day_bounds():
    when = datetime(2026, 10, 4, 23, 30, tzinfo=timezone.utc)
    start, end = local_day_bounds(when)
    local = when.astimezone()
    assert start == local.replace(hour=0, minute=0, second=0, microsecond=0)
    assert end - start == timedelta(days=1)
    assert start.tzinfo is not None
