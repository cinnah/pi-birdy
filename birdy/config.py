"""Load and validate birdy.toml."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    model: Path
    labels: Path
    threshold: float          # show a species at or above this
    rescue_threshold: float   # promote a gate-rejected frame to "bird" at or above this


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
    watchlist: tuple[str, ...] # live
    alert_new_species: bool    # live
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
class Config:
    site: SiteConfig
    camera: CameraConfig
    motion: MotionConfig
    daylight: DaylightConfig
    classifier: ClassifierConfig
    species: SpeciesConfig
    retention: RetentionConfig
    storage: StorageConfig
    web: WebConfig
    telegram: TelegramConfig
    publish: PublishConfig


def _size(v: Any, name: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) and not isinstance(i, bool) for i in v)):
        raise ConfigError(f"{name} must be [width, height]")
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
    rescue_threshold = _num(sp, "species", "rescue_threshold", 0.7, float)
    _check(0.0 <= rescue_threshold <= 1.0, "species.rescue_threshold must be between 0 and 1")
    _check(rescue_threshold >= species_threshold, "species.rescue_threshold must be >= species.threshold")

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
    tg_watch = _str_list(tg, "telegram", "watchlist")
    tg_new = _bool(tg, "telegram", "alert_new_species", True)
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
            model=_path(base, sp, "species", "model", "models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"),
            labels=_path(base, sp, "species", "labels", "models/inat_bird_labels.txt"),
            threshold=species_threshold,
            rescue_threshold=rescue_threshold,
        ),
        retention=RetentionConfig(non_bird_days=non_bird_days),
        storage=StorageConfig(data_dir=_path(base, sto, "storage", "data_dir", "data")),
        web=WebConfig(host=str(web.get("host", "0.0.0.0")), port=port),
        telegram=TelegramConfig(
            enabled=tg_enabled, bot_token=tg_token, chat_id=tg_chat, base_url=tg_base,
            watchlist=tg_watch, alert_new_species=tg_new, cooldown_minutes=tg_cooldown,
        ),
        publish=PublishConfig(
            enabled=pub_enabled, host=pub_host, user=pub_user, remote_dir=pub_remote, site_id=pub_site,
            ssh_key=pub_key, min_confidence=pub_min, per_species_per_day=pub_n,
        ),
    )
