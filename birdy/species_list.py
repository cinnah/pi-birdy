"""The site's local species list: birds seen on iNaturalist near the Pi, used to restrict the
BioCLIP candidates. Stored at data/species/local.json; refreshed at service start when stale."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from birdy.species import Bank

log = logging.getLogger(__name__)

API = "https://api.inaturalist.org/v1/observations/species_counts"
USER_AGENT = "birdy (+https://birdy.akibeo.com)"
MIN_OBSERVATIONS = 3
MIN_SPECIES = 20          # fewer -> retry once at 4x radius
MAX_AGE = timedelta(days=90)
PER_PAGE = 500
MAX_PAGES = 20           # per radius; a safety cap on paging


@dataclass(frozen=True)
class LocalList:
    fetched_at: datetime
    lat: float
    lon: float
    radius_km: int
    species: list[dict]    # {id, latin, common, count}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        payload = {**asdict(self), "fetched_at": self.fetched_at.isoformat()}
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> LocalList | None:
        try:
            d = json.loads(Path(path).read_text())
            fetched_at = datetime.fromisoformat(d["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            return cls(fetched_at, float(d["lat"]), float(d["lon"]), int(d["radius_km"]), list(d["species"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def ids(self) -> frozenset[int]:
        return frozenset(int(s["id"]) for s in self.species)


def _http_get(lat: float, lon: float, radius_km: int, page: int) -> dict:
    q = urllib.parse.urlencode({"lat": lat, "lng": lon, "radius": radius_km, "taxon_id": 3,
                                "quality_grade": "research", "locale": "en", "per_page": PER_PAGE, "page": page})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def parse_species_counts(pages: Iterable[dict]) -> list[dict]:
    out = []
    for page in pages:
        for r in page.get("results", []):
            t = r.get("taxon") or {}
            if t.get("rank") != "species" or int(r.get("count", 0)) < MIN_OBSERVATIONS:
                continue
            out.append({"id": int(t["id"]), "latin": t["name"], "common": t.get("preferred_common_name") or t["name"], "count": int(r["count"])})
    return out


def fetch_species_counts(lat: float, lon: float, radius_km: int, http: Callable = _http_get, sleep: Callable = time.sleep) -> list[dict]:
    """All pages for the radius (<= 1 request/s); retried once at 4x radius when fewer than MIN_SPECIES come back."""
    def pages(radius):
        got, page = [], 1
        while True:
            data = http(lat, lon, radius, page)
            got.append(data)
            if len(data.get("results", [])) < PER_PAGE:
                return got
            if page >= MAX_PAGES:
                log.warning("species list: stopped paging at %d pages within %d km", MAX_PAGES, radius)
                return got
            page += 1
            sleep(1.0)
    rows = parse_species_counts(pages(radius_km))
    if len(rows) < MIN_SPECIES:
        sleep(1.0)
        rows = parse_species_counts(pages(radius_km * 4))
    return rows


def needs_refresh(current: LocalList | None, lat: float, lon: float, radius_km: int, now: datetime) -> bool:
    if current is None:
        return True
    moved = abs(current.lat - lat) > 1e-6 or abs(current.lon - lon) > 1e-6 or current.radius_km != radius_km
    return moved or now - current.fetched_at > MAX_AGE


class SpeciesList:
    """Holds the current list and turns it + include/exclude into candidate ids for the bank.
    `version` bumps on every reload so the classifier knows to re-apply the candidates."""

    def __init__(self, data_dir: Path, bank: Bank) -> None:
        self.path = Path(data_dir) / "species" / "local.json"
        self.bank = bank
        self.version = 0
        self._lock = threading.Lock()
        self._current: LocalList | None = None
        self._thread: threading.Thread | None = None
        self._warned: set[str] = set()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._current = LocalList.load(self.path)
            self.version += 1

    def current(self) -> LocalList | None:
        with self._lock:
            return self._current

    def unknown_names(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if self.bank.resolve(n) is None]

    def candidates(self, species_cfg) -> frozenset[int] | None:
        """Local ids + include − exclude; None when there is no list and nothing is configured (whole bank)."""
        cur = self.current()
        if cur is None and not species_cfg.include and not species_cfg.exclude:
            return None
        ids = set(cur.ids()) if cur is not None else set(int(i) for i in self.bank.ids)
        for name in species_cfg.include:
            tid = self.bank.resolve(name)
            if tid is None:
                self._warn(name)
            else:
                ids.add(tid)
        for name in species_cfg.exclude:
            tid = self.bank.resolve(name)
            if tid is None:
                self._warn(name)
            else:
                ids.discard(tid)
        return frozenset(ids)

    def _warn(self, name: str) -> None:
        if name not in self._warned:
            log.warning("species.include/exclude: unknown name %r (not in the bank); ignored", name)
            self._warned.add(name)

    def refresh_if_needed(self, lat: float | None, lon: float | None, radius_km: int, force: bool = False) -> bool:
        """Fetch when the file is missing, stale or for another place (or `force`); True if a new list was saved."""
        if lat is None or lon is None:
            return False
        try:
            if not force and not needs_refresh(self.current(), lat, lon, radius_km, datetime.now(timezone.utc)):
                return False
            rows = fetch_species_counts(lat, lon, radius_km)
            if len(rows) < MIN_SPECIES:
                log.warning("species list: only %d species within %d km; keeping the current list", len(rows), radius_km)
                return False
            LocalList(datetime.now(timezone.utc), lat, lon, radius_km, rows).save(self.path)
            self.reload()
        except Exception as e:
            log.warning("species list refresh failed (%s); keeping the current list", e)
            return False
        log.info("species list: %d species within %d km", len(rows), radius_km)
        return True

    def start_refresh(self, lat: float | None, lon: float | None, radius_km: int) -> None:
        """One-shot background refresh at service start."""
        self._thread = threading.Thread(target=self.refresh_if_needed, args=(lat, lon, radius_km), name="species-list", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
