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
"""

FILTERS = {
    "all": "1 = 1",
    "birds": "is_bird = 1",
    "non-birds": "is_bird = 0",
}


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
        cv2.imwrite(str(out), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return rel_path

    def add_capture(self, captured_at: datetime, rel_path: Path, motion_score: int) -> int:
        thumb = self._make_thumbnail(rel_path)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO captures (captured_at, path, thumb_path, motion_score, status) VALUES (?, ?, ?, ?, 'pending')",
                (_to_db_time(captured_at), str(rel_path), None if thumb is None else str(thumb), motion_score),
            )
            return int(cur.lastrowid)

    def update_classification(
        self, capture_id: int, is_bird: bool, confidence: float, species: str | None, species_confidence: float | None
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ?, status = 'done' WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, capture_id),
            )

    def mark_error(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET status = 'error' WHERE id = ?", (capture_id,))

    def get(self, capture_id: int) -> Capture | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return None if row is None else _row_to_capture(row)

    def list_captures(self, filter: str = "all", limit: int = 50, offset: int = 0) -> list[Capture]:
        where = FILTERS[filter]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM captures WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def count(self, filter: str = "all") -> int:
        where = FILTERS[filter]
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM captures WHERE {where}").fetchone()[0])

    def pending(self) -> list[Capture]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM captures WHERE status = 'pending' ORDER BY id").fetchall()
        return [_row_to_capture(r) for r in rows]

    def last_capture_at(self) -> datetime | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(captured_at) AS t FROM captures").fetchone()
        return None if row["t"] is None else _from_db_time(row["t"])

    # --- maintenance ------------------------------------------------------

    def purge_non_birds(self, older_than_days: int, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        cutoff = _to_db_time(now - timedelta(days=older_than_days))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, thumb_path FROM captures WHERE is_bird = 0 AND captured_at < ?", (cutoff,)
            ).fetchall()
        deleted = 0
        for row in rows:
            for p in (self.photos_dir / row["path"], self.thumbs_dir / row["thumb_path"] if row["thumb_path"] else None):
                if p is not None:
                    p.unlink(missing_ok=True)
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM captures WHERE id = ?", (row["id"],))
            deleted += 1
        return deleted

    def disk_free_bytes(self) -> int:
        return shutil.disk_usage(self.data_dir).free

    def close(self) -> None:
        with self._lock:
            self._conn.close()
