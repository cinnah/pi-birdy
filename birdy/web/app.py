"""LAN web page over the store: capture grid, status JSON, and the settings page."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from birdy.config import ConfigError
from birdy.live import LiveConfig
from birdy.settings import FIELD_KEYS, parse_settings_form, save_settings
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
    live: LiveConfig,
    config_path: Path,
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
        pts = np.array(live.get().motion.roi, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=png.tobytes(), media_type="image/png")

    @app.get("/frame.jpg")
    def frame_jpg():
        frame = state.lores_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=jpg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def settings_context(
        form: dict | None = None,
        error: tuple[str | None, str] | None = None,
        saved: bool = False,
    ) -> dict:
        cfg = live.get()
        values = (
            form
            if form is not None
            else {
                "roi": [list(p) for p in cfg.motion.roi],
                "pixel_threshold": cfg.motion.pixel_threshold,
                "min_area": cfg.motion.min_area,
                "background_alpha": cfg.motion.background_alpha,
                "cooldown_seconds": cfg.motion.cooldown_seconds,
                "classifier_threshold": cfg.classifier.threshold,
                "lat": "" if cfg.site.lat is None else cfg.site.lat,
                "lon": "" if cfg.site.lon is None else cfg.site.lon,
                "margin_minutes": cfg.daylight.margin_minutes,
                "non_bird_days": cfg.retention.non_bird_days,
            }
        )
        if isinstance(values["roi"], str):
            # re-rendering a failed POST: keep the raw string the browser sent
            roi_json = values["roi"]
        else:
            roi_json = json.dumps(values["roi"])
        return {
            "v": values,
            "roi_json": roi_json,
            "lores_w": cfg.camera.lores_size[0],
            "lores_h": cfg.camera.lores_size[1],
            "readonly": {
                "camera.main_size": list(cfg.camera.main_size),
                "camera.lores_size": list(cfg.camera.lores_size),
                "camera.jpeg_quality": cfg.camera.jpeg_quality,
                "web.host": cfg.web.host,
                "web.port": cfg.web.port,
                "storage.data_dir": str(cfg.storage.data_dir),
                "classifier.model": str(cfg.classifier.model),
            },
            "error_field": None if error is None else error[0],
            "error_message": None if error is None else error[1],
            "saved": saved,
            "version": live.version,
            "status": status_payload(),
        }

    @app.get("/settings", response_class=HTMLResponse)
    def settings_get(request: Request, saved: int = 0):
        return templates.TemplateResponse(request, "settings.html", settings_context(saved=bool(saved)))

    @app.post("/settings")
    async def settings_post(request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        try:
            patch = parse_settings_form(form)
            new = save_settings(config_path, live.get(), patch)
        except ConfigError as e:
            msg = str(e)
            field = next((f for f, key in FIELD_KEYS.items() if msg.startswith(key)), None)
            ctx = settings_context(form=form, error=(field, msg))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=422)
        except OSError as e:
            ctx = settings_context(form=form, error=(None, f"could not write {config_path}: {e}"))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=500)
        live.replace(new)
        return RedirectResponse("/settings?saved=1", status_code=303)

    return app
