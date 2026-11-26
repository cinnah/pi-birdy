"""Publishing: which bird photos go to the public gallery, the local mirror, the manifest and the rsync push.

`rank_day` is pure. `Publisher` (further down) owns the DB state changes, the mirror folder
`data/publish/` and a worker thread that pushes it to the server.
"""

from __future__ import annotations

import dataclasses
import fcntl
import functools
import json
import logging
import os
import shlex
import shutil
import socket
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
RSYNC_TIMEOUT = 1800.0  # seconds; wall-clock cap on the whole run (--timeout=120 and ConnectTimeout=20 bound stalls)

Runner = Callable[[list[str]], int]


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except FileNotFoundError:
        return False


class _UnreadableOriginal(Exception):
    """The full-res JPEG cannot be decoded (zero-byte power-cut leftover, deleted by hand)."""


class _MirrorWriteError(Exception):
    """The mirror copy could not be written (disk full, permissions); transient, retried next run."""


def rsync_command(mirror_dir: Path, cfg: PublishConfig) -> list[str]:
    """Trailing slash on the source: sync the folder's contents, not the folder itself.

    --delay-updates/--delete-after: every file lands in one pass at the end, so the server never
    serves a manifest that lists photos not yet uploaded, and a timeout leaves it consistent.
    The cross-process `.lock` file and any half-written `*.tmp*` file never leave the Pi.
    """
    return [
        "rsync", "-az", "--delete", "--delete-after", "--delay-updates", "--timeout=120",
        "--exclude", ".lock", "--exclude", "*.tmp*",
        "-e", _ssh(cfg), f"{mirror_dir}/", f"{cfg.user}@{cfg.host}:{cfg.remote_dir}",
    ]


def _ssh(cfg: PublishConfig) -> str:
    return (
        f"ssh -i {shlex.quote(str(cfg.ssh_key))} -o BatchMode=yes -o ConnectTimeout=20"
        " -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    )


def public_key(cfg: PublishConfig) -> str | None:
    """The publish key's public half (one line), or None when no key has been made yet."""
    try:
        return Path(str(cfg.ssh_key) + ".pub").read_text().strip() or None
    except OSError:
        return None


def authorized_keys_line(cfg: PublishConfig) -> str | None:
    """What the gallery's keeper adds to the server for this feeder: the key, pinned to its folder."""
    key = public_key(cfg)
    return None if key is None else f'command="/usr/bin/rrsync /srv/birdy/sites/{cfg.site_id}",restrict {key}'


def create_publish_key(cfg: PublishConfig, run: Callable = subprocess.run) -> None:
    """Make the publish key pair if there is none (what `deploy/install.sh` does). Raises OSError
    or RuntimeError with ssh-keygen's message; never overwrites an existing key."""
    key = Path(cfg.ssh_key)
    if key.exists():
        return
    key.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    proc = run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-C", f"birdy-publish@{socket.gethostname()}"],
               capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ssh-keygen failed").strip().splitlines()[-1])


def check_connection(cfg: PublishConfig, run: Callable = subprocess.run) -> tuple[bool, str]:
    """Can this Pi reach its folder on the gallery server with its key? (ok, one line to show).
    Lists the folder; nothing is written."""
    cmd = ["rsync", "--list-only", "--timeout=20", "-e", _ssh(cfg), f"{cfg.user}@{cfg.host}:{cfg.remote_dir}"]
    try:
        proc = run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return False, "no answer from the server within 45 s"
    except OSError as e:
        return False, f"rsync could not be run: {e}"
    if proc.returncode == 0:
        return True, f"connected to {cfg.host}"
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip() and not ln.startswith("rsync error")]
    return False, (lines[0] if lines else f"rsync exit {proc.returncode}")


def _default_runner(cmd: list[str], timeout: float = RSYNC_TIMEOUT) -> int:
    try:
        return subprocess.call(cmd, stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("publish: rsync timed out after %.0f s", timeout)
        return -1
    except OSError as e:  # rsync/ssh not installed
        log.error("cannot run rsync: %s", e)
        return -2


class Publisher:
    """Decides what is published (DB), keeps the mirror folder and pushes it to the server.

    `evaluate`/`set_manual` are called from the pipeline and web threads; `run_once` from the
    worker thread (or `birdy publish` on the CLI). `_lock` serialises the read-rank-write cycle,
    `_run_lock` keeps two runs in this process from racing on the mirror folder, and an
    `flock` on `mirror_dir/.lock` does the same across processes (service vs `birdy publish`).
    """

    def __init__(
        self,
        live: LiveConfig,
        store: Store,
        mirror_dir: Path,
        runner: Runner | None = None,
        debounce: float = 120.0,
        max_interval: float = 900.0,
        rsync_timeout: float = RSYNC_TIMEOUT,
    ) -> None:
        self._live = live
        self._store = store
        self.mirror_dir = Path(mirror_dir)
        self.photos_dir = self.mirror_dir / "photos"
        self.thumbs_dir = self.mirror_dir / "thumbs"
        self._runner = runner or functools.partial(_default_runner, timeout=rsync_timeout)
        self._debounce = debounce
        self._max_interval = max_interval
        self._idle = 2.0          # how often the worker looks at the switch while publishing is off
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._last_published_ids: set[int] | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)

    @property
    def enabled(self) -> bool:
        """`publish.enabled` right now: a live switch on /settings. Off = nothing is ranked or pushed."""
        return self._live.get().publish.enabled

    # --- decisions -----------------------------------------------------------

    def evaluate(self, capture: Capture | None) -> bool:
        """Re-rank the species/day of `capture`; returns True if any published state changed."""
        if capture is None or not (capture.is_bird and capture.species and capture.status == "done"):
            return False
        return self.evaluate_day(capture.species, capture.captured_at)

    def evaluate_day(self, species: str, when: datetime) -> bool:
        """Re-rank one species on the local day of `when`; callers guard what counts as a candidate.

        Returns True if any published state changed. Does nothing while publishing is off: a photo
        ranked then would carry a publish state (and be kept by retention) for a gallery nobody feeds.
        """
        if not self.enabled:
            return False
        with self._lock:
            changed = self._rank(species, when)
        if changed:
            self.notify()
        return changed

    def evaluate_all(self) -> int:
        """Re-rank every species/day in the history; returns how many days changed."""
        days = self._store.species_days()
        changed = 0
        for species, when in days:
            with self._lock:
                if self._rank(species, when):
                    changed += 1
        if changed:
            self.notify()
        log.info("publish: ranked %d species-days, %d changed", len(days), changed)
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

    def forget(self, capture: Capture | None) -> None:
        """A capture was deleted: re-rank its species/day (it may have held a slot) and sync soon.

        The deleted row is already gone from the DB, so the next mirror sync drops its web copy and
        the server drops it on the following rsync (`--delete`).
        """
        if capture is None:
            return
        if capture.is_bird and capture.species and capture.status == "done":
            self.evaluate_day(capture.species, capture.captured_at)
        self.notify()

    def notify(self) -> None:
        """Ask the worker to sync soon (debounced)."""
        self._wake.set()

    # --- mirror + manifest + push -----------------------------------------------

    def run_once(self, now: datetime | None = None) -> bool:
        """Mirror, manifest, rsync. Returns False when rsync failed (the next tick retries), another
        run holds the lock, or `stop()` was called during the mirror build (nothing is pushed then)."""
        if not self._run_lock.acquire(blocking=False):
            log.warning("publish: a run is already in progress; skipping")
            return False
        try:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.mirror_dir / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:  # BlockingIOError is an OSError
                    log.warning("publish: another process is publishing; skipping")
                    return False
                try:
                    return self._run_once(now)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        finally:
            self._run_lock.release()

    def _run_once(self, now: datetime | None) -> bool:
        cfg = self._live.get().publish
        photos = self._sync_mirror()
        if photos is None:
            log.info("publish: stopping; mirror build interrupted, nothing pushed")
            return False
        self._write_manifest(photos, now)
        rc = self._runner(rsync_command(self.mirror_dir, cfg))
        if rc != 0:
            log.warning("publish: rsync to %s@%s failed (exit %d); will retry", cfg.user, cfg.host, rc)
            return False
        ids = {c.id for c in photos}
        level = logging.INFO if ids != self._last_published_ids else logging.DEBUG
        self._last_published_ids = ids
        log.log(level, "publish: %d photos synced to %s", len(photos), cfg.host)
        return True

    def _sync_mirror(self) -> list[Capture] | None:
        """Make the mirror match the published set; returns the captures that are on it.

        Returns None as soon as `stop()` is called: a first build can mean hundreds of resizes,
        and the service closes the store right after `stop()` returns. Nothing is swept then;
        the next run carries on where this one left off.
        """
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        keep: set[str] = set()
        out: list[Capture] = []
        for c in self._store.published():
            if self._stop.is_set():
                return None
            rel = c.photo
            try:
                self._ensure_web_copy(c, rel)
            except _UnreadableOriginal:
                with self._lock:  # don't interleave with a concurrent re-rank of this day
                    self._store.set_published(c.id, "hidden")
                    if c.species and c.status == "done":
                        self._rank(c.species, c.captured_at)  # the next-best photo takes the freed slot
                log.warning("publish: cannot read %s; hiding capture %d", rel, c.id)
                self.notify()
                continue
            except _MirrorWriteError as e:
                log.warning("publish: cannot write web copy of %s (%s); will retry", rel, e)
                continue
            if c.web_path != rel:
                self._store.set_web_path(c.id, rel)
                c = dataclasses.replace(c, web_path=rel)
            keep.add(rel)
            out.append(c)
        for root in (self.photos_dir, self.thumbs_dir):
            for f in root.rglob("*.jpg"):
                if f.relative_to(root).as_posix() not in keep:
                    f.unlink(missing_ok=True)
            for d in sorted((d for d in root.rglob("*") if d.is_dir()), reverse=True):  # children before parents
                try:
                    d.rmdir()  # only succeeds when empty
                except OSError:
                    pass
        # Files first, then the DB: a web_path pointing at a missing file is repaired on the next run anyway.
        for cid, _ in self._store.stale_web_paths():
            self._store.set_web_path(cid, None)
        return out

    def _ensure_web_copy(self, c: Capture, rel: str) -> None:
        """Web-sized JPEG (longest side <= WEB_MAX_SIDE, never upscaled) plus the store's thumb, copied as is.

        Both live at `rel` under photos/ and thumbs/: the thumb's mirror path equals `web_path` by
        construction (the store mirrors thumb paths on photo paths too). Missing or empty files are
        (re)made: an empty one is what a power cut left before `_write_mirror_file` fsynced.
        """
        photo = self.photos_dir / rel
        thumb = self.thumbs_dir / rel
        src_thumb = None if c.thumb_path is None else self._store.thumbs_dir / c.thumb_path
        if not _nonempty(photo):
            img = cv2.imread(str(self._store.photos_dir / rel))
            if img is None:
                raise _UnreadableOriginal(rel)
            h, w = img.shape[:2]
            scale = WEB_MAX_SIDE / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            self._write_mirror_file(photo, lambda tmp: cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY]))
        if not _nonempty(thumb) and src_thumb is not None and src_thumb.exists():
            self._write_mirror_file(thumb, lambda tmp: shutil.copyfile(src_thumb, tmp))

    @staticmethod
    def _write_mirror_file(target: Path, write: Callable[[Path], object]) -> None:
        """Write through a temp name in the same directory, then `os.replace`: a disk-full `imwrite`
        (which returns False after writing some bytes), a failed copy or a power cut never leaves a
        truncated file at `target` that a later run would accept. The temp name keeps the .jpg suffix
        so OpenCV picks the JPEG encoder; a leftover temp is removed by the `*.jpg` sweep anyway.
        The temp is fsynced before the rename: without it ext4 can commit the rename of a new file
        before its data, and a power cut leaves a 0-byte file (seen 2026-11-11).
        """
        tmp = target.with_name(target.stem + ".tmp.jpg")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if write(tmp) is False:  # cv2.imwrite reports failure by its return value, not an exception
                raise _MirrorWriteError(f"write {target} failed")
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except OSError as e:
            raise _MirrorWriteError(str(e)) from e
        finally:
            tmp.unlink(missing_ok=True)

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
        return {
            "version": MANIFEST_VERSION,
            "generated_at": now.isoformat(timespec="seconds"),
            "site": {
                "id": cfg.publish.site_id,
                "name": cfg.site.name,
                "lat": None if cfg.site.lat is None else round(cfg.site.lat, 2),
                "lon": None if cfg.site.lon is None else round(cfg.site.lon, 2),
                "since": stats["first_capture"],
                "utc_offset_minutes": int(now.utcoffset().total_seconds() // 60),
            },
            "photos": [self._photo_entry(c) for c in photos],
            "stats": stats,
        }

    def _write_manifest(self, photos: list[Capture], now: datetime | None) -> None:
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        target = self.mirror_dir / "manifest.json"
        tmp = target.with_name(f"manifest.json.tmp.{os.getpid()}")  # unique per process
        try:
            tmp.write_text(json.dumps(self.build_manifest(photos, now), ensure_ascii=False, indent=1))
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)  # no-op after a successful replace

    # --- thread -----------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("publisher did not stop within %.1fs (rsync still running)", timeout)

    def _run(self) -> None:
        on = False          # was publishing on at the last look?
        seen_off = False    # has it been off since this thread started?
        last_run = 0.0
        while not self._stop.is_set():
            if not self.enabled:
                on, seen_off = False, True
                self._stop.wait(self._idle)
                continue
            if not on:
                on = True
                # Just switched on, or the first time ever (no mirror yet): rank the history, since
                # nothing was ranked while it was off. A plain restart with it on keeps its picks.
                if seen_off or not self.mirror_dir.exists():
                    try:
                        self.evaluate_all()
                    except Exception:
                        log.exception("publish: ranking history failed")
                self._wake.clear()  # the first run below happens right away anyway
                last_run = time.monotonic() - self._max_interval  # reconcile now
            remaining = last_run + self._max_interval - time.monotonic()
            woke = self._wake.wait(max(0.0, remaining))
            if self._stop.is_set():
                break
            if woke:
                # Coalesce a burst (a backfill sweep can publish many photos in a few seconds).
                if self._stop.wait(self._debounce):
                    break
            self._wake.clear()
            if not self.enabled:      # switched off while waiting: push nothing
                continue
            try:
                self.run_once()
            except Exception:
                log.exception("publish run failed")
            last_run = time.monotonic()
