"""Decide whether capture should run based on sunrise/sunset."""

from __future__ import annotations

from datetime import datetime, timedelta

from astral import Observer
from astral.sun import sun


class DaylightGate:
    def __init__(self, lat: float | None, lon: float | None, margin_minutes: int) -> None:
        self.enabled = lat is not None and lon is not None
        self._observer = Observer(latitude=lat, longitude=lon) if self.enabled else None
        self._margin = timedelta(minutes=margin_minutes)

    def _window(self, now: datetime, day_offset: int = 0) -> tuple[datetime, datetime]:
        assert self._observer is not None
        date = (now + timedelta(days=day_offset)).date()
        s = sun(self._observer, date=date, tzinfo=now.tzinfo)
        return s["sunrise"] - self._margin, s["sunset"] + self._margin

    def is_active(self, now: datetime) -> bool:
        """`now` must be timezone-aware local time, e.g. `datetime.now().astimezone()`;
        the day's window is computed for the date in `now`'s timezone."""
        if not self.enabled:
            return True
        start, end = self._window(now)
        return start <= now < end

    def next_change(self, now: datetime) -> datetime | None:
        """When the active/inactive state next flips, or None if the gate is disabled."""
        if not self.enabled:
            return None
        start, end = self._window(now)
        if now < start:
            return start
        if now < end:
            return end
        return self._window(now, day_offset=1)[0]
