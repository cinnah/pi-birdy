"""Publishing: which bird photos go to the public gallery, the local mirror, the manifest and the rsync push.

`rank_day` is pure. `Publisher` (further down) owns the DB state changes, the mirror folder
`data/publish/` and a worker thread that pushes it to the server.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from birdy.config import Config, PublishConfig
from birdy.live import LiveConfig
from birdy.store import Capture, Store

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
    # Both bounds carry `when`'s fixed offset, so on a DST-change day they are off by one hour at the edges (accepted).
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
    eligible.sort(key=lambda c: (-c.species_confidence, c.id))
    for i, c in enumerate(eligible):
        desired[c.id] = "auto" if i < settings.per_species_per_day else None
    return desired


WEB_MAX_SIDE = 1600
WEB_JPEG_QUALITY = 85
MANIFEST_VERSION = 1
RSYNC_TIMEOUT = 300.0  # seconds; rsync's own --timeout only covers I/O stalls, this bounds the whole run

Runner = Callable[[list[str]], int]


def rsync_command(mirror_dir: Path, cfg: PublishConfig) -> list[str]:
    """Trailing slash on the source: sync the folder's contents, not the folder itself."""
    ssh = f"ssh -i {shlex.quote(str(cfg.ssh_key))} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    return ["rsync", "-az", "--delete", "--timeout=120", "-e", ssh, f"{mirror_dir}/", f"{cfg.user}@{cfg.host}:{cfg.remote_dir}"]


def _default_runner(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd, stdin=subprocess.DEVNULL, timeout=RSYNC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return -1
    except OSError as e:  # rsync/ssh not installed
        log.error("cannot run rsync: %s", e)
        return -2


class Publisher:
    """Decides what is published (DB), keeps the mirror folder and pushes it to the server.

    `evaluate`/`set_manual` are called from the pipeline and web threads; `run_once` from the
    worker thread (or `birdy publish` on the CLI). `_lock` serialises the read-rank-write cycle.
    """

    def __init__(
        self,
        live: LiveConfig,
        store: Store,
        mirror_dir: Path,
        runner: Runner | None = None,
        debounce: float = 120.0,
        max_interval: float = 900.0,
    ) -> None:
        self._live = live
        self._store = store
        self.mirror_dir = Path(mirror_dir)
        self.photos_dir = self.mirror_dir / "photos"
        self.thumbs_dir = self.mirror_dir / "thumbs"
        self._runner = runner or _default_runner
        self._debounce = debounce
        self._max_interval = max_interval
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)

    # --- decisions -----------------------------------------------------------

    def evaluate(self, capture: Capture | None) -> bool:
        """Re-rank the species/day of `capture`; returns True if any published state changed."""
        if capture is None or not (capture.is_bird and capture.species and capture.status == "done"):
            return False
        with self._lock:
            changed = self._rank(capture.species, capture.captured_at)
        if changed:
            self.notify()
        return changed

    def _rank(self, species: str, when: datetime) -> bool:
        """Apply `rank_day` to one species/day; caller holds `_lock`."""
        start, end = local_day_bounds(when)
        candidates = self._store.species_day_candidates(species, start, end)
        desired = rank_day(candidates, self._store.manual_states(), PublishSettings.from_config(self._live.get()))
        changed = False
        for c in candidates:
            if desired[c.id] != c.published:
                self._store.set_published(c.id, desired[c.id])
                changed = True
        return changed

    def set_manual(self, capture_id: int, state: str | None) -> None:
        """'forced' | 'hidden' | None (back to the rule). Raises KeyError for an unknown capture."""
        if state not in (None, "forced", "hidden"):
            raise ValueError(f"bad manual state {state!r}")
        c = self._store.get(capture_id)
        if c is None:
            raise KeyError(capture_id)
        with self._lock:
            self._store.set_published(capture_id, state)
            if c.is_bird and c.species and c.status == "done":
                self._rank(c.species, c.captured_at)  # a freed or taken slot moves other photos
        self.notify()

    def notify(self) -> None:
        """Ask the worker to sync soon (debounced)."""
        self._wake.set()

    # --- mirror + manifest + push -----------------------------------------------

    def run_once(self, now: datetime | None = None) -> bool:
        """Mirror, manifest, rsync. Returns False when rsync failed (the next tick retries)."""
        cfg = self._live.get().publish
        photos = self._sync_mirror()
        self._write_manifest(photos, now)
        rc = self._runner(rsync_command(self.mirror_dir, cfg))
        if rc != 0:
            log.warning("publish: rsync to %s@%s failed (exit %d); will retry", cfg.user, cfg.host, rc)
            return False
        log.info("publish: %d photos synced to %s", len(photos), cfg.host)
        return True

    def _sync_mirror(self) -> list[Capture]:
        """Make the mirror match the published set; returns the captures that are on it."""
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        keep: set[str] = set()
        out: list[Capture] = []
        for c in self._store.published():
            if c.web_path is None or not (self.photos_dir / c.web_path).exists():
                if not self._make_web_copy(c):
                    with self._lock:  # don't interleave with a concurrent re-rank of this day
                        self._store.set_published(c.id, "hidden")
                    log.warning("publish: cannot read %s; hiding capture %d", c.path, c.id)
                    continue
                self._store.set_web_path(c.id, c.path)
                c = self._store.get(c.id)
            keep.add(c.web_path)
            out.append(c)
        for root in (self.photos_dir, self.thumbs_dir):
            for f in root.rglob("*.jpg"):
                if f.relative_to(root).as_posix() not in keep:
                    f.unlink()
        # Files first, then the DB: a web_path pointing at a missing file is repaired on the next run anyway.
        for cid, _ in self._store.stale_web_paths():
            self._store.set_web_path(cid, None)
        return out

    def _make_web_copy(self, c: Capture) -> bool:
        """Web-sized JPEG (longest side <= WEB_MAX_SIDE, never upscaled) plus the store's thumb, copied as is."""
        img = cv2.imread(str(self._store.photos_dir / c.path))
        if img is None:
            return False
        h, w = img.shape[:2]
        scale = WEB_MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out = self.photos_dir / c.path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY]):
            return False
        src_thumb = None if c.thumb_path is None else self._store.thumbs_dir / c.thumb_path
        if src_thumb is not None and src_thumb.exists():
            dst = self.thumbs_dir / c.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_thumb, dst)
        return True

    def _photo_entry(self, c: Capture) -> dict:
        thumb = f"thumbs/{c.web_path}" if (self.thumbs_dir / c.web_path).exists() else f"photos/{c.web_path}"
        return {
            "id": c.id,
            "captured_at": c.captured_at.astimezone().isoformat(timespec="seconds"),
            "species": c.species,
            "confidence": None if c.species_confidence is None else round(c.species_confidence, 3),
            "photo": f"photos/{c.web_path}",
            "thumb": thumb,
            "state": c.published,
        }

    def build_manifest(self, photos: list[Capture], now: datetime | None = None) -> dict:
        """The JSON the static gallery pages consume; its shape is a contract (MANIFEST_VERSION)."""
        cfg = self._live.get()
        now = (now or datetime.now(timezone.utc)).astimezone()
        stats = self._store.publish_stats(now)
        site = self._store.site() or {"name": cfg.site.name, "lat": cfg.site.lat, "lon": cfg.site.lon}
        return {
            "version": MANIFEST_VERSION,
            "generated_at": now.isoformat(timespec="seconds"),
            "site": {
                "id": cfg.publish.site_id,
                "name": site["name"],
                "lat": None if site["lat"] is None else round(site["lat"], 2),
                "lon": None if site["lon"] is None else round(site["lon"], 2),
                "since": stats["first_capture"],
                "utc_offset_minutes": int(now.utcoffset().total_seconds() // 60),
            },
            "photos": [self._photo_entry(c) for c in photos],
            "stats": stats,
        }

    def _write_manifest(self, photos: list[Capture], now: datetime | None) -> None:
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        target = self.mirror_dir / "manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.build_manifest(photos, now), ensure_ascii=False, indent=1))
        os.replace(tmp, target)

    # --- thread -----------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        last_run = time.monotonic() - self._max_interval  # first run right after start: reconcile
        while not self._stop.is_set():
            remaining = last_run + self._max_interval - time.monotonic()
            woke = self._wake.wait(max(0.0, remaining))
            if self._stop.is_set():
                break
            if woke:
                # Coalesce a burst (a backfill sweep can publish many photos in a few seconds).
                if self._stop.wait(self._debounce):
                    break
            self._wake.clear()
            try:
                self.run_once()
            except Exception:
                log.exception("publish run failed")
            last_run = time.monotonic()
