"""Load and validate birdy.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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
class Config:
    site: SiteConfig
    camera: CameraConfig
    motion: MotionConfig
    daylight: DaylightConfig
    classifier: ClassifierConfig
    retention: RetentionConfig
    storage: StorageConfig
    web: WebConfig


def _size(v: Any, name: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) for i in v)):
        raise ConfigError(f"{name} must be [width, height]")
    return (v[0], v[1])


def _roi(v: Any) -> tuple[Point, ...]:
    if not isinstance(v, list) or len(v) < 3:
        raise ConfigError("motion.roi must be a list of at least 3 [x, y] points")
    pts = []
    for p in v:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) for i in p)):
            raise ConfigError("motion.roi points must be [x, y] integers")
        pts.append((p[0], p[1]))
    return tuple(pts)


def _path(base: Path, v: str) -> Path:
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
    lat, lon = site.get("lat"), site.get("lon")
    if (lat is None) != (lon is None):
        raise ConfigError("site.lat and site.lon must both be set or both omitted")

    cam = raw.get("camera", {})
    mot = raw.get("motion", {})
    if "roi" not in mot:
        raise ConfigError("motion.roi is required")
    day = raw.get("daylight", {})
    cls = raw.get("classifier", {})
    threshold = float(cls.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("classifier.threshold must be between 0 and 1")
    ret = raw.get("retention", {})
    sto = raw.get("storage", {})
    web = raw.get("web", {})

    return Config(
        site=SiteConfig(name=str(site["name"]), lat=None if lat is None else float(lat), lon=None if lon is None else float(lon)),
        camera=CameraConfig(
            main_size=_size(cam.get("main_size", [3280, 2464]), "camera.main_size"),
            lores_size=_size(cam.get("lores_size", [320, 240]), "camera.lores_size"),
            jpeg_quality=int(cam.get("jpeg_quality", 90)),
        ),
        motion=MotionConfig(
            roi=_roi(mot["roi"]),
            pixel_threshold=int(mot.get("pixel_threshold", 25)),
            min_area=int(mot.get("min_area", 150)),
            background_alpha=float(mot.get("background_alpha", 0.05)),
            cooldown_seconds=float(mot.get("cooldown_seconds", 3.0)),
        ),
        daylight=DaylightConfig(margin_minutes=int(day.get("margin_minutes", 30))),
        classifier=ClassifierConfig(
            model=_path(base, cls.get("model", "models/mobilenet_v2_1.0_224_quant.tflite")),
            labels=_path(base, cls.get("labels", "models/imagenet_labels.txt")),
            bird_classes=_path(base, cls.get("bird_classes", "models/imagenet_bird_classes.txt")),
            threshold=threshold,
        ),
        retention=RetentionConfig(non_bird_days=int(ret.get("non_bird_days", 7))),
        storage=StorageConfig(data_dir=_path(base, sto.get("data_dir", "data"))),
        web=WebConfig(host=str(web.get("host", "0.0.0.0")), port=int(web.get("port", 8000))),
    )
