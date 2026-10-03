"""Editable settings: form parsing, config serialisation and atomic save."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from birdy.config import Config, ConfigError, load_config

Point = tuple[int, int]

HEADER = "# Written by the birdy settings page. Comments live in birdy.toml.example.\n\n"

# form field -> config key (used for error messages and field highlighting)
FIELD_KEYS = {
    "roi": "motion.roi",
    "pixel_threshold": "motion.pixel_threshold",
    "min_area": "motion.min_area",
    "background_alpha": "motion.background_alpha",
    "cooldown_seconds": "motion.cooldown_seconds",
    "classifier_threshold": "classifier.threshold",
    "lat": "site.lat",
    "lon": "site.lon",
    "margin_minutes": "daylight.margin_minutes",
    "non_bird_days": "retention.non_bird_days",
}


@dataclass(frozen=True)
class SettingsPatch:
    """Exactly the keys the settings page may change."""

    roi: tuple[Point, ...]
    pixel_threshold: int
    min_area: int
    background_alpha: float
    cooldown_seconds: float
    classifier_threshold: float
    lat: float | None
    lon: float | None
    margin_minutes: int
    non_bird_days: int


def config_to_dict(config: Config) -> dict:
    """Every key of the config as plain TOML-serialisable values (paths absolute, no None)."""
    site: dict = {"name": config.site.name}
    if config.site.lat is not None and config.site.lon is not None:
        site["lat"] = config.site.lat
        site["lon"] = config.site.lon
    return {
        "site": site,
        "camera": {
            "main_size": list(config.camera.main_size),
            "lores_size": list(config.camera.lores_size),
            "jpeg_quality": config.camera.jpeg_quality,
        },
        "motion": {
            "roi": [list(p) for p in config.motion.roi],
            "pixel_threshold": config.motion.pixel_threshold,
            "min_area": config.motion.min_area,
            "background_alpha": config.motion.background_alpha,
            "cooldown_seconds": config.motion.cooldown_seconds,
        },
        "daylight": {"margin_minutes": config.daylight.margin_minutes},
        "classifier": {
            "model": str(config.classifier.model),
            "labels": str(config.classifier.labels),
            "bird_classes": str(config.classifier.bird_classes),
            "threshold": config.classifier.threshold,
        },
        "retention": {"non_bird_days": config.retention.non_bird_days},
        "storage": {"data_dir": str(config.storage.data_dir)},
        "web": {"host": config.web.host, "port": config.web.port},
    }


def apply_patch(data: dict, patch: SettingsPatch) -> dict:
    data["motion"].update(
        roi=[list(p) for p in patch.roi],
        pixel_threshold=patch.pixel_threshold,
        min_area=patch.min_area,
        background_alpha=patch.background_alpha,
        cooldown_seconds=patch.cooldown_seconds,
    )
    data["classifier"]["threshold"] = patch.classifier_threshold
    data["site"].pop("lat", None)
    data["site"].pop("lon", None)
    # Write whichever is given; load_config rejects one-without-the-other with a clear message.
    if patch.lat is not None:
        data["site"]["lat"] = patch.lat
    if patch.lon is not None:
        data["site"]["lon"] = patch.lon
    data["daylight"]["margin_minutes"] = patch.margin_minutes
    data["retention"]["non_bird_days"] = patch.non_bird_days
    return data


def save_settings(path: Path, current: Config, patch: SettingsPatch) -> Config:
    """Validate `current` + `patch`, write `path` atomically, return the new Config.

    On ConfigError nothing is written and the temp file is removed.
    """
    path = Path(path)
    data = apply_patch(config_to_dict(current), patch)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(HEADER + tomli_w.dumps(data))
    try:
        new = load_config(tmp)
    except ConfigError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return new


def _num(form: Mapping[str, str], field: str, cast: type, optional: bool = False):
    key = FIELD_KEYS[field]
    raw = form.get(field)
    if raw is None or raw.strip() == "":
        if optional:
            return None
        raise ConfigError(f"{key} is required")
    try:
        # int("7.5") raises, so fractional input for an int field is rejected rather than truncated.
        return cast(raw.strip())
    except ValueError:
        raise ConfigError(f"{key} must be a {cast.__name__}") from None


def _roi(form: Mapping[str, str]) -> tuple[Point, ...]:
    key = FIELD_KEYS["roi"]
    raw = form.get("roi", "")
    try:
        pts = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError(f"{key} must be JSON like [[x, y], ...]") from None
    if not isinstance(pts, list):
        raise ConfigError(f"{key} must be a list of [x, y] points")
    out = []
    for p in pts:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in p)):
            raise ConfigError(f"{key} points must be [x, y] integers")
        out.append((p[0], p[1]))
    return tuple(out)


def parse_settings_form(form: Mapping[str, str]) -> SettingsPatch:
    """Turn the POSTed form into a SettingsPatch; raises ConfigError naming the config key."""
    return SettingsPatch(
        roi=_roi(form),
        pixel_threshold=_num(form, "pixel_threshold", int),
        min_area=_num(form, "min_area", int),
        background_alpha=_num(form, "background_alpha", float),
        cooldown_seconds=_num(form, "cooldown_seconds", float),
        classifier_threshold=_num(form, "classifier_threshold", float),
        lat=_num(form, "lat", float, optional=True),
        lon=_num(form, "lon", float, optional=True),
        margin_minutes=_num(form, "margin_minutes", int),
        non_bird_days=_num(form, "non_bird_days", int),
    )
