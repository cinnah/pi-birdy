"""Publishing: which bird photos go to the public gallery, the local mirror, the manifest and the rsync push.

`rank_day` is pure. `Publisher` (further down) owns the DB state changes, the mirror folder
`data/publish/` and a worker thread that pushes it to the server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from birdy.config import Config
from birdy.store import Capture

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishSettings:
    """The live-tunable subset, read from live.get().publish on each evaluation."""

    min_confidence: float
    per_species_per_day: int

    @classmethod
    def from_config(cls, cfg: Config) -> "PublishSettings":
        return cls(cfg.publish.min_confidence, cfg.publish.per_species_per_day)


def local_day_bounds(when: datetime) -> tuple[datetime, datetime]:
    """[start, end) of the local calendar day containing `when` (the Pi's zone), tz-aware."""
    start = when.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def rank_day(candidates: list[Capture], manual: dict[int, str], settings: PublishSettings) -> dict[int, str | None]:
    """Desired `published` value for every candidate of one species on one local day.

    Manual states ('forced'/'hidden') are kept as they are and 'forced' does not use a slot.
    Of the rest, those at or above `min_confidence`, best confidence first (ties: lower id first),
    the first `per_species_per_day` get 'auto'; the others None.
    """
    desired: dict[int, str | None] = {}
    eligible: list[Capture] = []
    for c in candidates:
        m = manual.get(c.id)
        if m in ("forced", "hidden"):
            desired[c.id] = m
        elif c.species_confidence is not None and c.species_confidence >= settings.min_confidence:
            eligible.append(c)
        else:
            desired[c.id] = None
    eligible.sort(key=lambda c: (-(c.species_confidence or 0.0), c.id))
    for i, c in enumerate(eligible):
        desired[c.id] = "auto" if i < settings.per_species_per_day else None
    return desired
