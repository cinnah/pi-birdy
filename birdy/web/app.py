"""LAN web page over the store: capture grid, status JSON, species summary and the settings page."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from birdy.classifier import Prediction
from birdy.config import ConfigError
from birdy.live import LiveConfig
from birdy.notify import TelegramError
from birdy.settings import FIELD_KEYS, parse_settings_form, save_settings
from birdy.state import RuntimeState
from birdy.store import FILTERS, Store

PAGE_SIZE = 50
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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
    classifier: Any | None = None,
    notifier: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `classifier` is an object with `classify_image(img_rgb) -> (Prediction, top)`, normally a
    `TwoStageClassifier`; None disables `/classify`.

    `notifier` is an object with `send_test()`; normally a `TelegramNotifier`; None = alerts disabled.
    """
    app = FastAPI(title="birdy")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["local_time"] = _local_time

    app.mount("/photos", StaticFiles(directory=str(store.photos_dir)), name="photos")
    app.mount("/thumbs", StaticFiles(directory=str(store.thumbs_dir)), name="thumbs")

    def status_payload() -> dict:
        snap = state.snapshot()
        last = store.last_capture_at()
        ls = store.last_species()
        snap.update(
            queue_depth=queue_depth(),
            pending_count=store.count_pending(),
            disk_free_bytes=store.disk_free_bytes(),
            last_capture_at=None if last is None else last.isoformat(),
            last_species=None if ls is None else ls[0],
        )
        return snap

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, filter: str = "all", page: int = 1, species: str | None = None):
        if filter not in FILTERS:
            raise HTTPException(status_code=400, detail="filter must be one of all, birds, non-birds")
        summary = store.species_summary()
        if species == "":
            species = None
        if species is not None and species not in {r.species for r in summary}:
            raise HTTPException(status_code=400, detail="unknown species")
        total = store.count(filter, species=species)
        pages = max(1, -(-total // PAGE_SIZE))
        page = min(max(1, page), pages)
        captures = store.list_captures(filter, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE, species=species)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "captures": captures,
                "filter": filter,
                "page": page,
                "pages": pages,
                "total": total,
                "species": species,
                "species_rows": summary,
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

    def species_context(result: dict | None = None, error: str | None = None) -> dict:
        return {"rows": store.species_summary(), "result": result, "error": error}

    def _decode_and_classify(data: bytes) -> tuple[Prediction, list] | None:
        """Decode an uploaded image and run both models; None if the bytes aren't an image."""
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            return None
        if img is None:
            return None
        return classifier.classify_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @app.get("/species", response_class=HTMLResponse)
    def species_page(request: Request):
        return templates.TemplateResponse(request, "species.html", species_context())

    @app.post("/classify", response_class=HTMLResponse)
    async def classify_upload(request: Request):
        if classifier is None:
            raise HTTPException(status_code=503, detail="classifier not available")
        # Starlette spools the whole upload before the size check below (fine on a LAN-only service).
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            ctx = species_context(error="choose a photo first")
            return templates.TemplateResponse(request, "species.html", ctx, status_code=422)
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            ctx = species_context(error="choose a photo first")
            return templates.TemplateResponse(request, "species.html", ctx, status_code=422)
        if len(data) > MAX_UPLOAD_BYTES:
            ctx = species_context(error="photo larger than 10 MB")
            return templates.TemplateResponse(request, "species.html", ctx, status_code=413)
        # decode + inference can take a while on the Pi; keep the event loop free for /status etc.
        outcome = await run_in_threadpool(_decode_and_classify, data)
        if outcome is None:
            ctx = species_context(error="could not decode that file as an image")
            return templates.TemplateResponse(request, "species.html", ctx, status_code=422)
        pred, top = outcome
        result = {
            "is_bird": pred.is_bird,
            "confidence": pred.confidence,
            "species": pred.species,
            "top": [(label.common, label.latin, p) for label, p in top[:3]],
            "filename": upload.filename or "photo",
        }
        return templates.TemplateResponse(request, "species.html", species_context(result=result))

    def settings_context(
        form: dict | None = None,
        error: tuple[str | None, str] | None = None,
        saved: bool = False,
        test: tuple[bool, str] | None = None,
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
                "species_threshold": cfg.species.threshold,
                "species_rescue_threshold": cfg.species.rescue_threshold,
                "lat": "" if cfg.site.lat is None else cfg.site.lat,
                "lon": "" if cfg.site.lon is None else cfg.site.lon,
                "margin_minutes": cfg.daylight.margin_minutes,
                "non_bird_days": cfg.retention.non_bird_days,
                "watchlist": "\n".join(cfg.telegram.watchlist),
                "alert_new_species": cfg.telegram.alert_new_species,
                "cooldown_minutes": cfg.telegram.cooldown_minutes,
                "publish_min_confidence": cfg.publish.min_confidence,
                "per_species_per_day": cfg.publish.per_species_per_day,
            }
        )
        roi = values.get("roi", "")
        if isinstance(roi, str):
            # re-rendering a failed POST: keep the raw string the browser sent
            roi_json = roi
        else:
            roi_json = json.dumps(roi)
        return {
            "v": values,
            "roi_json": roi_json,
            # what is actually in effect, regardless of what the (possibly rejected) form sent
            "saved_roi_json": json.dumps([list(p) for p in cfg.motion.roi]),
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
                "species.model": str(cfg.species.model),
                "telegram.enabled": cfg.telegram.enabled,
                "telegram.chat_id": cfg.telegram.chat_id or "–",
                # never the token itself
                "telegram.bot_token": "set" if cfg.telegram.bot_token else "not set",
                "telegram.base_url": cfg.telegram.base_url,
                "publish.enabled": cfg.publish.enabled,
                "publish.host": cfg.publish.host or "–",
                "publish.user": cfg.publish.user,
                "publish.remote_dir": cfg.publish.remote_dir,
                "publish.site_id": cfg.publish.site_id,
            },
            "species_seen": [r.species for r in store.species_summary()],
            "alerts_enabled": notifier is not None,
            "test_ok": None if test is None else test[0],
            "test_result": None if test is None else test[1],
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

    @app.post("/settings/test-alert", response_class=HTMLResponse)
    async def settings_test_alert(request: Request):
        if notifier is None:
            ctx = settings_context(
                test=(
                    False,
                    "alerts are disabled: set [telegram] enabled = true (with bot_token and chat_id) "
                    "in birdy.toml and restart",
                )
            )
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=503)
        try:
            await run_in_threadpool(notifier.send_test)
        except TelegramError as e:
            ctx = settings_context(test=(False, f"Telegram error: {e.description}"))
            # TelegramError also wraps transport failures (DNS, timeouts, connection resets)
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=502)
        ctx = settings_context(test=(True, "test alert sent — check Telegram"))
        return templates.TemplateResponse(request, "settings.html", ctx)

    return app
