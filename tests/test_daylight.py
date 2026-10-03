from datetime import datetime, timezone

from birdy.daylight import DaylightGate

LISBON = dict(lat=38.72, lon=-9.14)


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_disabled_without_coordinates():
    g = DaylightGate(lat=None, lon=None, margin_minutes=30)
    assert g.enabled is False
    assert g.is_active(utc(2026, 6, 21, 2)) is True
    assert g.next_change(utc(2026, 6, 21, 2)) is None


def test_midday_is_active():
    g = DaylightGate(**LISBON, margin_minutes=30)
    assert g.is_active(utc(2026, 6, 21, 13)) is True


def test_night_is_inactive():
    g = DaylightGate(**LISBON, margin_minutes=30)
    assert g.is_active(utc(2026, 6, 21, 2)) is False
    assert g.is_active(utc(2026, 6, 21, 23)) is False


def test_margin_extends_window():
    # Lisbon sunrise on 2026-06-21 is ~05:12 UTC. 04:55 UTC is inside a 30-min margin, outside a 0-min one.
    assert DaylightGate(**LISBON, margin_minutes=30).is_active(utc(2026, 6, 21, 4, 55)) is True
    assert DaylightGate(**LISBON, margin_minutes=0).is_active(utc(2026, 6, 21, 4, 55)) is False


def test_next_change_before_sunrise_is_today_start():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 2))
    assert nxt.date() == utc(2026, 6, 21, 2).date() and 4 <= nxt.hour <= 6


def test_next_change_during_day_is_today_end():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 13))
    assert nxt.hour >= 19


def test_next_change_after_sunset_is_tomorrow_start():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 23))
    assert nxt.day == 22 and 4 <= nxt.hour <= 6
