"""Load and validate birdy.toml."""

from __future__ import annotations

import logging
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

Point = tuple[int, int]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SiteConfig:
    name: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class CameraConfig:
    main_size: tuple[int, int]
    lores_size: tuple[int, int]
    jpeg_quality: int


@dataclass(frozen=True)
class MotionConfig:
    roi: tuple[Point, ...]
    pixel_threshold: int
    min_area: int
    background_alpha: float
    cooldown_seconds: float


@dataclass(frozen=True)
class DaylightConfig:
    margin_minutes: int


@dataclass(frozen=True)
class ClassifierConfig:
    model: Path
    labels: Path
    bird_classes: Path
    threshold: float


@dataclass(frozen=True)
class SpeciesConfig:
    model: Path                     # restart-only; BioCLIP 2 image tower (int8 ONNX)
    bank: Path                      # restart-only; name embeddings for every iNat bird species
    threshold: float                # name a bird at or above this (live)
    region_radius_km: int           # iNaturalist radius for the local species list; refetch at next start
    include: tuple[str, ...]        # names added to the local list, latin or common (live)
    exclude: tuple[str, ...]        # names dropped from it (live)


@dataclass(frozen=True)
class DetectorConfig:
    model: Path               # restart-only
    labels: Path              # restart-only
    threshold: float          # a COCO "bird" detection at or above this makes the photo a bird (live)


@dataclass(frozen=True)
class VisitsConfig:
    gap_seconds: int          # photos of one bird within this gap form a visit (live)


@dataclass(frozen=True)
class RetentionConfig:
    non_bird_days: int


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool              # restart-only
    bot_token: str = field(repr=False)  # restart-only, secret
    chat_id: str               # restart-only
    base_url: str              # restart-only; link in the alert caption
    no_alert: tuple[str, ...]  # live; species that never alert (every other species does)
    cooldown_minutes: int      # live


@dataclass(frozen=True)
class PublishConfig:
    enabled: bool              # restart-only; the publisher thread is only built when true
    host: str                  # restart-only; gallery server
    user: str                  # restart-only; SSH user on the server
    remote_dir: str            # restart-only; rsync target as the server key sees it ("/" with rrsync)
    site_id: str               # restart-only; folder under sites/ and id in sites.json
    ssh_key: Path              # restart-only; private key (~ expanded)
    min_confidence: float      # live
    per_species_per_day: int   # live


@dataclass(frozen=True)
class DslrConfig:
    enabled: bool                               # live; the second camera (gphoto2 over USB), a switch on /settings
    region: tuple[float, float, float, float]   # live; x, y, w, h fractions of the DSLR frame the detector searches
    daily_cap: int                              # live; maximum DSLR shots per local day
    auto_shutter: bool                          # live; pick the DSLR's shutter speed from the Pi camera's light reading


@dataclass(frozen=True)
class Config:
    site: SiteConfig
    camera: CameraConfig
    motion: MotionConfig
    daylight: DaylightConfig
    classifier: ClassifierConfig
    species: SpeciesConfig
    detector: DetectorConfig
    visits: VisitsConfig
    retention: RetentionConfig
    storage: StorageConfig
    web: WebConfig
    telegram: TelegramConfig
    publish: PublishConfig
    dslr: DslrConfig


def _size(v: Any, name: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) and not isinstance(i, bool) for i in v)):
        raise ConfigError(f"{name} must be [width, height]")
    if v[0] <= 0 or v[1] <= 0:
        raise ConfigError(f"{name} must be positive")
    return (v[0], v[1])


def _roi(v: Any) -> tuple[Point, ...]:
    if not isinstance(v, list) or len(v) < 3:
        raise ConfigError("motion.roi must be a list of at least 3 [x, y] points")
    pts = []
    for p in v:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) and not isinstance(i, bool) for i in p)):
            raise ConfigError("motion.roi points must be [x, y] integers")
        pts.append((p[0], p[1]))
    return tuple(pts)


def _region(v: Any, name: str) -> tuple[float, float, float, float]:
    if not (isinstance(v, list) and len(v) == 4
            and all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in v)):
        raise ConfigError(f"{name} must be [x, y, w, h] as fractions of the frame")
    x, y, w, h = (float(i) for i in v)
    ok = all(math.isfinite(i) for i in (x, y, w, h)) and x >= 0 and y >= 0 and w > 0 and h > 0 \
        and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
    if not ok:
        raise ConfigError(f"{name} must lie inside the frame: values 0..1, w and h > 0, x + w <= 1, y + h <= 1")
    return (x, y, w, h)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


_KIND = {int: "an integer", float: "a number"}


def _num(section: dict[str, Any], name: str, key: str, default: Any, cast: type) -> Any:
    v = section.get(key, default)
    err = f"{name}.{key} must be {_KIND[cast]}"
    if isinstance(v, bool):
        raise ConfigError(err)
    if cast is int and isinstance(v, float) and not v.is_integer():
        raise ConfigError(err)
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise ConfigError(err) from None
    if cast is float and not math.isfinite(v):
        raise ConfigError(f"{name}.{key} must be a finite number")
    return v


def _bool(section: dict[str, Any], name: str, key: str, default: bool) -> bool:
    v = section.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"{name}.{key} must be true or false")
    return v


def _str(section: dict[str, Any], name: str, key: str, default: str) -> str:
    v = section.get(key, default)
    if not isinstance(v, str):
        raise ConfigError(f"{name}.{key} must be a string")
    return v


def _str_list(section: dict[str, Any], name: str, key: str) -> tuple[str, ...]:
    v = section.get(key, [])
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise ConfigError(f"{name}.{key} must be a list of strings")
    return tuple(s.strip() for s in v if s.strip())


def _plain_ssh_name(v: str) -> bool:
    """Safe to pass to ssh/rsync as `user@host`: no option injection, no embedded user/port."""
    return bool(v) and not v.startswith("-") and re.search(r"[\s@:]", v) is None


def _path(base: Path, section: dict[str, Any], name: str, key: str, default: str) -> Path:
    v = section.get(key, default)
    if not isinstance(v, str):
        raise ConfigError(f"{name}.{key} must be a path string")
    p = Path(v)
    return p if p.is_absolute() else base / p


SPECIES_MODEL_DEFAULT = "models/bioclip2-vision-int8.onnx"


def _species_model(base: Path, sp: dict[str, Any]) -> Path:
    """species.model; an old TFLite path (from before BioCLIP) falls back to the default with a warning."""
    p = _path(base, sp, "species", "model", SPECIES_MODEL_DEFAULT)
    if p.suffix.lower() == ".tflite":
        default = base / SPECIES_MODEL_DEFAULT
        log.warning("species.model %s is the old TFLite model; using %s (edit birdy.toml or save the settings page)", p, default)
        return default
    return p


def load_config(path: Path) -> Config:
    path = Path(path)
    base = path.parent.resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    site = raw.get("site", {})
    if "name" not in site:
        raise ConfigError("site.name is required")
    if (site.get("lat") is None) != (site.get("lon") is None):
        raise ConfigError("site.lat and site.lon must both be set or both omitted")
    lat = lon = None
    if site.get("lat") is not None:
        lat = _num(site, "site", "lat", None, float)
        lon = _num(site, "site", "lon", None, float)
        _check(-90 <= lat <= 90, "site.lat must be between -90 and 90")
        _check(-180 <= lon <= 180, "site.lon must be between -180 and 180")

    cam = raw.get("camera", {})
    main_size = _size(cam.get("main_size", [3280, 2464]), "camera.main_size")
    lores_size = _size(cam.get("lores_size", [320, 240]), "camera.lores_size")
    (mw, mh), (lw0, lh0) = main_size, lores_size
    _check(
        abs(mw / mh - lw0 / lh0) <= 0.01 * (mw / mh),
        "camera.main_size and camera.lores_size must have the same aspect ratio (the motion crop is drawn in lores coordinates)",
    )
    jpeg_quality = _num(cam, "camera", "jpeg_quality", 90, int)
    _check(1 <= jpeg_quality <= 100, "camera.jpeg_quality must be between 1 and 100")

    mot = raw.get("motion", {})
    if "roi" not in mot:
        raise ConfigError("motion.roi is required")
    roi = _roi(mot["roi"])
    lw, lh = lores_size
    for x, y in roi:
        _check(0 <= x < lw and 0 <= y < lh, f"motion.roi point [{x}, {y}] is outside camera.lores_size {lw}x{lh}")
    pixel_threshold = _num(mot, "motion", "pixel_threshold", 25, int)
    _check(pixel_threshold >= 0, "motion.pixel_threshold must be >= 0")
    min_area = _num(mot, "motion", "min_area", 150, int)
    _check(min_area >= 0, "motion.min_area must be >= 0")
    background_alpha = _num(mot, "motion", "background_alpha", 0.05, float)
    _check(0 < background_alpha <= 1, "motion.background_alpha must be in (0, 1]")
    cooldown_seconds = _num(mot, "motion", "cooldown_seconds", 3.0, float)
    _check(cooldown_seconds >= 0, "motion.cooldown_seconds must be >= 0")

    day = raw.get("daylight", {})
    margin_minutes = _num(day, "daylight", "margin_minutes", 30, int)
    _check(margin_minutes >= 0, "daylight.margin_minutes must be >= 0")

    cls = raw.get("classifier", {})
    threshold = _num(cls, "classifier", "threshold", 0.5, float)
    _check(0.0 <= threshold <= 1.0, "classifier.threshold must be between 0 and 1")

    sp = raw.get("species", {})
    species_threshold = _num(sp, "species", "threshold", 0.5, float)
    _check(0.0 <= species_threshold <= 1.0, "species.threshold must be between 0 and 1")
    region_radius_km = _num(sp, "species", "region_radius_km", 50, int)
    _check(1 <= region_radius_km <= 500, "species.region_radius_km must be between 1 and 500")
    sp_include = _str_list(sp, "species", "include")
    sp_exclude = _str_list(sp, "species", "exclude")

    det = raw.get("detector", {})
    detector_threshold = _num(det, "detector", "threshold", 0.2, float)
    _check(0.0 <= detector_threshold <= 1.0, "detector.threshold must be between 0 and 1")

    vis = raw.get("visits", {})
    gap = _num(vis, "visits", "gap_seconds", 60, int)
    _check(10 <= gap <= 600, "visits.gap_seconds must be between 10 and 600")

    ret = raw.get("retention", {})
    non_bird_days = _num(ret, "retention", "non_bird_days", 7, int)
    _check(non_bird_days >= 1, "retention.non_bird_days must be >= 1")

    sto = raw.get("storage", {})
    web = raw.get("web", {})
    port = _num(web, "web", "port", 8000, int)
    _check(1 <= port <= 65535, "web.port must be between 1 and 65535")

    tg = raw.get("telegram", {})
    tg_enabled = _bool(tg, "telegram", "enabled", False)
    tg_token = _str(tg, "telegram", "bot_token", "")
    if isinstance(tg.get("chat_id"), int) and not isinstance(tg.get("chat_id"), bool):
        tg = {**tg, "chat_id": str(tg["chat_id"])}  # allow an unquoted numeric chat id
    tg_chat = _str(tg, "telegram", "chat_id", "")
    tg_base = _str(tg, "telegram", "base_url", "http://birdy.local:8000").rstrip("/")
    # the old watchlist / alert_new_species keys are ignored: every species alerts unless listed here
    tg_no_alert = _str_list(tg, "telegram", "no_alert")
    tg_cooldown = _num(tg, "telegram", "cooldown_minutes", 30, int)
    _check(tg_cooldown >= 0, "telegram.cooldown_minutes must be >= 0")
    if tg_enabled:
        _check(bool(tg_token), "telegram.bot_token is required when telegram.enabled = true")
        _check(bool(tg_chat), "telegram.chat_id is required when telegram.enabled = true")

    pub = raw.get("publish", {})
    pub_enabled = _bool(pub, "publish", "enabled", False)
    pub_host = _str(pub, "publish", "host", "").strip()
    pub_user = _str(pub, "publish", "user", "birdy").strip()
    pub_remote = _str(pub, "publish", "remote_dir", "/").strip()
    pub_site = _str(pub, "publish", "site_id", "home")
    _check(re.fullmatch(r"[a-z0-9-]{1,32}", pub_site) is not None, "publish.site_id must match [a-z0-9-]{1,32}")
    pub_key = Path(_str(pub, "publish", "ssh_key", "~/.ssh/birdy_publish")).expanduser()
    pub_key = pub_key if pub_key.is_absolute() else base / pub_key
    pub_min = _num(pub, "publish", "min_confidence", 0.6, float)
    _check(0.0 <= pub_min <= 1.0, "publish.min_confidence must be between 0 and 1")
    pub_n = _num(pub, "publish", "per_species_per_day", 3, int)
    _check(1 <= pub_n <= 20, "publish.per_species_per_day must be between 1 and 20")
    if pub_enabled:
        _check(bool(pub_host), "publish.host is required when publish.enabled = true")
        _check(bool(pub_user), "publish.user is required when publish.enabled = true")
        _check(_plain_ssh_name(pub_host), "publish.host must be a plain hostname or IP")
        _check(_plain_ssh_name(pub_user), "publish.user must be a plain user name")
        _check(pub_remote.startswith("/"), "publish.remote_dir must be an absolute path")

    ds = raw.get("dslr", {})
    dslr_enabled = _bool(ds, "dslr", "enabled", False)
    dslr_region = _region(ds.get("region", [0.0, 0.0, 1.0, 1.0]), "dslr.region")
    dslr_cap = _num(ds, "dslr", "daily_cap", 200, int)
    dslr_auto = _bool(ds, "dslr", "auto_shutter", False)
    _check(1 <= dslr_cap <= 2000, "dslr.daily_cap must be between 1 and 2000")

    return Config(
        site=SiteConfig(name=str(site["name"]), lat=lat, lon=lon),
        camera=CameraConfig(main_size=main_size, lores_size=lores_size, jpeg_quality=jpeg_quality),
        motion=MotionConfig(
            roi=roi,
            pixel_threshold=pixel_threshold,
            min_area=min_area,
            background_alpha=background_alpha,
            cooldown_seconds=cooldown_seconds,
        ),
        daylight=DaylightConfig(margin_minutes=margin_minutes),
        classifier=ClassifierConfig(
            model=_path(base, cls, "classifier", "model", "models/mobilenet_v2_1.0_224_quant.tflite"),
            labels=_path(base, cls, "classifier", "labels", "models/imagenet_labels.txt"),
            bird_classes=_path(base, cls, "classifier", "bird_classes", "models/imagenet_bird_classes.txt"),
            threshold=threshold,
        ),
        species=SpeciesConfig(
            model=_species_model(base, sp),
            bank=_path(base, sp, "species", "bank", "models/bioclip2-birds.npz"),
            threshold=species_threshold,
            region_radius_km=region_radius_km,
            include=sp_include,
            exclude=sp_exclude,
        ),
        detector=DetectorConfig(
            model=_path(base, det, "detector", "model", "models/efficientdet_lite0_320_ptq.tflite"),
            labels=_path(base, det, "detector", "labels", "models/coco_labels.txt"),
            threshold=detector_threshold,
        ),
        visits=VisitsConfig(gap_seconds=gap),
        retention=RetentionConfig(non_bird_days=non_bird_days),
        storage=StorageConfig(data_dir=_path(base, sto, "storage", "data_dir", "data")),
        web=WebConfig(host=str(web.get("host", "0.0.0.0")), port=port),
        telegram=TelegramConfig(
            enabled=tg_enabled, bot_token=tg_token, chat_id=tg_chat, base_url=tg_base,
            no_alert=tg_no_alert, cooldown_minutes=tg_cooldown,
        ),
        publish=PublishConfig(
            enabled=pub_enabled, host=pub_host, user=pub_user, remote_dir=pub_remote, site_id=pub_site,
            ssh_key=pub_key, min_confidence=pub_min, per_species_per_day=pub_n,
        ),
        dslr=DslrConfig(enabled=dslr_enabled, region=dslr_region, daily_cap=dslr_cap, auto_shutter=dslr_auto),
    )
