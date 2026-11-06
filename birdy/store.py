"""SQLite + on-disk photo storage. One Store instance is shared by all threads."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

THUMB_WIDTH = 400
PUBLISHED_STATES = (None, "auto", "forced", "hidden")

SCHEMA = """
CREATE TABLE IF NOT EXISTS site (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  name TEXT NOT NULL,
  lat REAL,
  lon REAL
);
CREATE TABLE IF NOT EXISTS captures (
  id INTEGER PRIMARY KEY,
  captured_at TEXT NOT NULL,
  path TEXT NOT NULL,
  thumb_path TEXT,
  motion_score INTEGER NOT NULL,
  is_bird INTEGER,
  confidence REAL,
  species TEXT,
  species_confidence REAL,
  status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS captures_captured_at ON captures(captured_at DESC);
CREATE INDEX IF NOT EXISTS captures_is_bird ON captures(is_bird, captured_at DESC);
CREATE INDEX IF NOT EXISTS captures_status ON captures(status);
CREATE INDEX IF NOT EXISTS captures_species ON captures(species, captured_at DESC);
"""

FILTERS = {
    "all": "1 = 1",
    "birds": "is_bird = 1",
    "non-birds": "is_bird = 0",
    "published": "published IN ('auto', 'forced')",
}

_BACKLOG_WHERE = "status = 'done' AND (species_checked = 0 OR detector_checked = 0) AND manual_at IS NULL"
_BACKLOG_COUNT = f"SELECT COUNT(*) FROM captures WHERE {_BACKLOG_WHERE}"
_BACKLOG_SELECT = f"SELECT * FROM captures WHERE {_BACKLOG_WHERE} ORDER BY captured_at DESC, id DESC LIMIT ?"


@dataclass(frozen=True)
class Capture:
    id: int
    captured_at: datetime
    path: str
    thumb_path: str | None
    motion_score: int
    is_bird: bool | None
    confidence: float | None
    species: str | None
    species_confidence: float | None
    status: str
    species_checked: bool
    published: str | None = None   # None | 'auto' | 'forced' | 'hidden'
    web_path: str | None = None    # relative path of the web-sized copy under data/publish/photos
    # Motion box at the trigger frame as fractions (x, y, w, h) of the lores frame; None = unknown.
    motion_box: tuple[float, float, float, float] | None = None
    # The user's correction (None = none) and, once corrected, what the model had said.
    manual_is_bird: bool | None = None
    manual_at: datetime | None = None
    model_is_bird: bool | None = None
    model_species: str | None = None
    model_species_confidence: float | None = None
    # Best COCO "bird" detection at classification time: score (0 = none, None = not run yet)
    # and its box as fractions (x, y, w, h) of the classified crop.
    det_confidence: float | None = None
    det_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SpeciesRow:
    species: str
    count: int
    first_seen: datetime
    last_seen: datetime
    best_capture_id: int
    best_path: str
    best_thumb_path: str | None


def _to_db_time(dt: datetime) -> str:
    """Store as UTC ISO 8601 so string comparison == chronological comparison."""
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _from_db_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _row_to_capture(row: sqlite3.Row) -> Capture:
    return Capture(
        id=row["id"],
        captured_at=_from_db_time(row["captured_at"]),
        path=row["path"],
        thumb_path=row["thumb_path"],
        motion_score=row["motion_score"],
        is_bird=None if row["is_bird"] is None else bool(row["is_bird"]),
        confidence=row["confidence"],
        species=row["species"],
        species_confidence=row["species_confidence"],
        status=row["status"],
        species_checked=bool(row["species_checked"]),
        published=row["published"],
        web_path=row["web_path"],
        motion_box=None if row["motion_x"] is None else (row["motion_x"], row["motion_y"], row["motion_w"], row["motion_h"]),
        manual_is_bird=None if row["manual_is_bird"] is None else bool(row["manual_is_bird"]),
        manual_at=None if row["manual_at"] is None else _from_db_time(row["manual_at"]),
        model_is_bird=None if row["model_is_bird"] is None else bool(row["model_is_bird"]),
        model_species=row["model_species"],
        model_species_confidence=row["model_species_confidence"],
        det_confidence=row["det_confidence"],
        det_box=None if row["det_x"] is None else (row["det_x"], row["det_y"], row["det_w"], row["det_h"]),
    )


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.photos_dir = self.data_dir / "photos"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "birdy.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(captures)")}
            for name, decl in (
                ("species_checked", "INTEGER NOT NULL DEFAULT 0"),
                ("published", "TEXT"),
                ("web_path", "TEXT"),
                ("motion_x", "REAL"),
                ("motion_y", "REAL"),
                ("motion_w", "REAL"),
                ("motion_h", "REAL"),
                # The model's verdict, kept here once a person corrects the row (the final
                # is_bird/species columns then hold the correction).
                ("model_is_bird", "INTEGER"),
                ("model_confidence", "REAL"),
                ("model_species", "TEXT"),
                ("model_species_confidence", "REAL"),
                ("manual_is_bird", "INTEGER"),
                ("manual_species", "TEXT"),
                ("manual_at", "TEXT"),
                ("det_confidence", "REAL"),
                ("det_x", "REAL"),
                ("det_y", "REAL"),
                ("det_w", "REAL"),
                ("det_h", "REAL"),
                ("detector_checked", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in cols:
                    try:
                        self._conn.execute(f"ALTER TABLE captures ADD COLUMN {name} {decl}")
                    except sqlite3.OperationalError as e:
                        # Another process (service vs `birdy purge`) migrated between our check and ALTER.
                        if "duplicate column" not in str(e):
                            raise
            self._conn.execute("CREATE INDEX IF NOT EXISTS captures_published ON captures(published)")

    # --- site -------------------------------------------------------------

    def set_site(self, name: str, lat: float | None, lon: float | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO site (id, name, lat, lon) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name, lat = excluded.lat, lon = excluded.lon",
                (name, lat, lon),
            )

    def site(self) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT name, lat, lon FROM site WHERE id = 1").fetchone()
        return None if row is None else {"name": row["name"], "lat": row["lat"], "lon": row["lon"]}

    # --- captures ---------------------------------------------------------

    def photo_path_for(self, captured_at: datetime) -> tuple[Path, Path]:
        """Return (absolute path, path relative to photos_dir). Creates the day directory."""
        local = captured_at.astimezone()
        rel = Path(local.strftime("%Y/%m/%d")) / f"{local.strftime('%H%M%S')}_{local.microsecond // 1000:03d}.jpg"
        abs_path = self.photos_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path, rel

    def _make_thumbnail(self, rel_path: Path) -> Path | None:
        img = cv2.imread(str(self.photos_dir / rel_path))
        if img is None:
            log.warning("could not read %s for thumbnail", rel_path)
            return None
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
        out = self.thumbs_dir / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            log.warning("could not write thumbnail %s", out)
            return None
        return rel_path

    def add_capture(
        self,
        captured_at: datetime,
        rel_path: Path,
        motion_score: int,
        motion_box: tuple[float, float, float, float] | None = None,
    ) -> int:
        thumb = self._make_thumbnail(rel_path)
        mx, my, mw, mh = motion_box if motion_box is not None else (None, None, None, None)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO captures (captured_at, path, thumb_path, motion_score, status, motion_x, motion_y, motion_w, motion_h)"
                " VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (_to_db_time(captured_at), str(rel_path), None if thumb is None else str(thumb), motion_score, mx, my, mw, mh),
            )
            return int(cur.lastrowid)

    def update_classification(
        self,
        capture_id: int,
        is_bird: bool,
        confidence: float,
        species: str | None,
        species_confidence: float | None,
        det_confidence: float = 0.0,
        det_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        bx = (None, None, None, None) if det_box is None else det_box
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ?,"
                " det_confidence = ?, det_x = ?, det_y = ?, det_w = ?, det_h = ?,"
                " species_checked = 1, detector_checked = 1, status = 'done' WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, det_confidence, *bx, capture_id),
            )

    def mark_not_bird(self, capture_id: int) -> Capture | None:
        """A person says there is no bird: the row becomes a non-bird and leaves the gallery.

        The model's verdict is copied to `model_*` on the first correction only, so "back to model"
        can restore it. Returns the row as it was before, or None for an unknown id.
        """
        now = _to_db_time(datetime.now(timezone.utc))
        with self._lock, self._conn:
            row = self._conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
            if row is None:
                return None
            if row["manual_at"] is None:
                self._conn.execute(
                    "UPDATE captures SET model_is_bird = is_bird, model_confidence = confidence,"
                    " model_species = species, model_species_confidence = species_confidence WHERE id = ?",
                    (capture_id,),
                )
            self._conn.execute(
                "UPDATE captures SET manual_is_bird = 0, manual_species = NULL, manual_at = ?,"
                " is_bird = 0, species = NULL, species_confidence = NULL, published = NULL WHERE id = ?",
                (now, capture_id),
            )
        return _row_to_capture(row)

    def clear_correction(self, capture_id: int) -> Capture | None:
        """Back to the model's verdict (from `model_*`); returns the restored row, or None if unknown."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET is_bird = model_is_bird, confidence = model_confidence,"
                " species = model_species, species_confidence = model_species_confidence,"
                " manual_is_bird = NULL, manual_species = NULL, manual_at = NULL"
                " WHERE id = ? AND manual_at IS NOT NULL",
                (capture_id,),
            )
        return self.get(capture_id)

    def mark_error(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET status = 'error' WHERE id = ?", (capture_id,))

    def get(self, capture_id: int) -> Capture | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return None if row is None else _row_to_capture(row)

    def _where(self, filter: str, species: str | None) -> tuple[str, list]:
        where, params = FILTERS[filter], []
        if species is not None:
            where += " AND species = ? AND is_bird = 1"
            params.append(species)
        return where, params

    def list_captures(
        self, filter: str = "all", limit: int = 50, offset: int = 0, species: str | None = None
    ) -> list[Capture]:
        where, params = self._where(filter, species)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM captures WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def count(self, filter: str = "all", species: str | None = None) -> int:
        where, params = self._where(filter, species)
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM captures WHERE {where}", params).fetchone()[0])

    def pending(self) -> list[Capture]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM captures WHERE status = 'pending' ORDER BY id").fetchall()
        return [_row_to_capture(r) for r in rows]

    def count_pending(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM captures WHERE status = 'pending'").fetchone()[0])

    def last_capture_at(self) -> datetime | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(captured_at) AS t FROM captures").fetchone()
        return None if row["t"] is None else _from_db_time(row["t"])

    def count_classification_backlog(self) -> int:
        with self._lock:
            return int(self._conn.execute(_BACKLOG_COUNT).fetchone()[0])

    def classification_backlog(self, limit: int = 50) -> list[Capture]:
        """Classified rows a newer model hasn't seen (species or detector), newest first.
        Corrected rows never: re-running the models must not undo a person's correction."""
        with self._lock:
            rows = self._conn.execute(_BACKLOG_SELECT, (limit,)).fetchall()
        return [_row_to_capture(r) for r in rows]

    def mark_checked(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET species_checked = 1, detector_checked = 1 WHERE id = ?", (capture_id,))

    def species_summary(self) -> list[SpeciesRow]:
        with self._lock:
            groups = self._conn.execute(
                "SELECT species, COUNT(*) AS n, MIN(captured_at) AS first_seen, MAX(captured_at) AS last_seen"
                " FROM captures WHERE species IS NOT NULL AND is_bird = 1 GROUP BY species ORDER BY last_seen DESC"
            ).fetchall()
            out = []
            for g in groups:
                best = self._conn.execute(
                    "SELECT id, path, thumb_path FROM captures WHERE species = ? AND is_bird = 1"
                    " ORDER BY species_confidence DESC, captured_at DESC LIMIT 1",
                    (g["species"],),
                ).fetchone()
                out.append(
                    SpeciesRow(
                        species=g["species"],
                        count=int(g["n"]),
                        first_seen=_from_db_time(g["first_seen"]),
                        last_seen=_from_db_time(g["last_seen"]),
                        best_capture_id=int(best["id"]),
                        best_path=best["path"],
                        best_thumb_path=best["thumb_path"],
                    )
                )
        return out

    def last_species(self) -> tuple[str, datetime] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT species, captured_at FROM captures WHERE species IS NOT NULL AND is_bird = 1"
                " ORDER BY captured_at DESC LIMIT 1"
            ).fetchone()
        return None if row is None else (row["species"], _from_db_time(row["captured_at"]))

    # --- publishing -------------------------------------------------------

    def set_published(self, capture_id: int, state: str | None) -> None:
        """Set the publish state (one of PUBLISHED_STATES); an unknown id is a silent no-op."""
        if state not in PUBLISHED_STATES:
            raise ValueError(f"bad published state {state!r}")
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET published = ? WHERE id = ?", (state, capture_id))

    def set_web_path(self, capture_id: int, web_path: str | None) -> None:
        """Record (or clear) the web-sized copy's relative path; an unknown id is a silent no-op."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET web_path = ? WHERE id = ?", (web_path, capture_id))

    def species_day_candidates(self, species: str, start: datetime, end: datetime) -> list[Capture]:
        """Classified birds of one species captured in [start, end), best confidence first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM captures WHERE is_bird = 1 AND status = 'done' AND species = ?"
                " AND captured_at >= ? AND captured_at < ? ORDER BY species_confidence DESC, id ASC",
                (species, _to_db_time(start), _to_db_time(end)),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def species_days(self) -> list[tuple[str, datetime]]:
        """One (species, capture time) per species per local day, over classified birds with a species."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT species, MIN(captured_at) AS t FROM captures"
                " WHERE is_bird = 1 AND status = 'done' AND species IS NOT NULL"
                " GROUP BY species, date(captured_at, 'localtime') ORDER BY t, species"
            ).fetchall()
        return [(r["species"], _from_db_time(r["t"])) for r in rows]

    def published(self) -> list[Capture]:
        """Captures on the gallery ('auto' or 'forced'), newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM captures WHERE published IN ('auto', 'forced') ORDER BY captured_at DESC, id DESC"
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def manual_states(self) -> dict[int, str]:
        """Manual overrides as {capture id: 'forced' | 'hidden'}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, published FROM captures WHERE published IN ('forced', 'hidden')"
            ).fetchall()
        return {int(r["id"]): r["published"] for r in rows}

    def stale_web_paths(self) -> list[tuple[int, str]]:
        """Captures that still have a web copy but are no longer published."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, web_path FROM captures WHERE web_path IS NOT NULL"
                " AND (published IS NULL OR published = 'hidden') ORDER BY id"
            ).fetchall()
        return [(int(r["id"]), r["web_path"]) for r in rows]

    def publish_stats(self, now: datetime | None = None, days: int = 30) -> dict:
        """Aggregate numbers for the gallery dashboard, over ALL bird captures (published or not).

        Times are local (the Pi's zone); `per_day` has `days` entries ending today, `per_hour` 24;
        each species carries `days`: {local day: count} for the days in that window it was seen.
        """
        now = (now or datetime.now(timezone.utc)).astimezone()
        with self._lock:
            rows = self._conn.execute("SELECT captured_at, species FROM captures WHERE is_bird = 1").fetchall()
        today = now.date()
        per_day = {today - timedelta(days=i): 0 for i in range(days)}
        per_hour = [0] * 24
        by_species: dict[str, list] = {}  # name -> [count, first, last, {day: count} within the window]
        first = last = None
        for r in rows:
            t = _from_db_time(r["captured_at"]).astimezone()
            d = t.date()
            if d in per_day:
                per_day[d] += 1
            per_hour[t.hour] += 1
            first = t if first is None or t < first else first
            last = t if last is None or t > last else last
            if r["species"]:
                s = by_species.setdefault(r["species"], [0, t, t, {}])
                s[0] += 1
                s[1] = min(s[1], t)
                s[2] = max(s[2], t)
                if d in per_day:
                    s[3][d.isoformat()] = s[3].get(d.isoformat(), 0) + 1
        species = [
            {"name": n, "count": c, "first": f.date().isoformat(), "last": l.isoformat(timespec="seconds"),
             "days": dict(sorted(ds.items()))}
            for n, (c, f, l, ds) in by_species.items()
        ]
        species.sort(key=lambda s: (-s["count"], s["name"]))
        return {
            "total_birds": len(rows),
            "species_count": len(by_species),
            "first_capture": None if first is None else first.date().isoformat(),
            "last_capture_at": None if last is None else last.isoformat(timespec="seconds"),
            "species": species,
            "per_day": [{"day": d.isoformat(), "birds": per_day[d]} for d in sorted(per_day)],
            "per_hour": per_hour,
        }

    # --- maintenance ------------------------------------------------------

    def purge_non_birds(self, older_than_days: int, now: datetime | None = None) -> int:
        """Delete non-bird and errored captures older than the cutoff (rows + files).

        `pending` rows are kept (the sweep will classify them); birds are never purged, and neither
        is a row with any `published` state (forced, hidden or auto: someone cares about it) or a
        row a person corrected (a labelled example, kept for training).
        """
        now = now or datetime.now(timezone.utc)
        cutoff = _to_db_time(now - timedelta(days=older_than_days))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, thumb_path FROM captures"
                " WHERE (is_bird = 0 OR status = 'error') AND published IS NULL AND manual_at IS NULL AND captured_at < ?",
                (cutoff,),
            ).fetchall()
        for row in rows:
            self._delete_row_and_files(row["id"], row["path"], row["thumb_path"])
        return len(rows)

    def delete_capture(self, capture_id: int) -> Capture | None:
        """Delete one capture (row + photo + thumbnail), whatever it is; returns it, or None if unknown.

        Used by the grid's Delete button. Bird photos are never auto-deleted, but a person may.
        """
        c = self.get(capture_id)
        if c is None:
            return None
        self._delete_row_and_files(c.id, c.path, c.thumb_path)
        return c

    def delete_non_birds(self, capture_ids: list[int]) -> list[Capture]:
        """Delete those of `capture_ids` that are non-birds or errors; birds and pending rows are kept.

        Used by the grid's bulk delete on the non-birds view; returns what was deleted.
        """
        out = []
        for cid in capture_ids:
            c = self.get(cid)
            if c is not None and (c.is_bird is False or c.status == "error"):
                self._delete_row_and_files(c.id, c.path, c.thumb_path)
                out.append(c)
        return out

    def _delete_row_and_files(self, capture_id: int, path: str, thumb_path: str | None) -> None:
        # Row first: an orphan file is harmless, a row without its file is a broken tile.
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        for p in (self.photos_dir / path, self.thumbs_dir / thumb_path if thumb_path else None):
            if p is not None:
                p.unlink(missing_ok=True)

    def disk_free_bytes(self) -> int:
        return shutil.disk_usage(self.data_dir).free

    def close(self) -> None:
        with self._lock:
            self._conn.close()
