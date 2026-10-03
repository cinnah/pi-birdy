"""Read-only LAN web page over the store."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from birdy.state import RuntimeState
from birdy.store import FILTERS, Store

PAGE_SIZE = 50


def _local_time(dt: datetime | str | None) -> str:
    """Render a datetime (or its ISO string) in the Pi's local time.

    Naive datetimes are assumed to be UTC, matching how the store records times.
    None renders as a dash.
    """
    if dt is None:
        return "–"
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def create_app(
    store: Store,
    state: RuntimeState,
    roi: Sequence[tuple[int, int]],
    queue_depth: Callable[[], int] = lambda: 0,
) -> FastAPI:
    app = FastAPI(title="birdy")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["local_time"] = _local_time

    app.mount("/photos", StaticFiles(directory=str(store.photos_dir)), name="photos")
    app.mount("/thumbs", StaticFiles(directory=str(store.thumbs_dir)), name="thumbs")

    def status_payload() -> dict:
        snap = state.snapshot()
        last = store.last_capture_at()
        snap.update(
            queue_depth=queue_depth(),
            pending_count=store.count_pending(),
            disk_free_bytes=store.disk_free_bytes(),
            last_capture_at=None if last is None else last.isoformat(),
        )
        return snap

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, filter: str = "all", page: int = 1):
        if filter not in FILTERS:
            raise HTTPException(status_code=400, detail="filter must be one of all, birds, non-birds")
        total = store.count(filter)
        pages = max(1, -(-total // PAGE_SIZE))
        page = min(max(1, page), pages)
        captures = store.list_captures(filter, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "captures": captures,
                "filter": filter,
                "page": page,
                "pages": pages,
                "total": total,
                "status": status_payload(),
            },
        )

    @app.get("/status")
    def status():
        return status_payload()

    @app.get("/roi.png")
    def roi_png():
        frame = state.lores_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=png.tobytes(), media_type="image/png")

    return app
