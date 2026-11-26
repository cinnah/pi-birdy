"""Editable settings: form parsing, config serialisation and atomic save."""

from __future__ import annotations

import json
import os
import shutil
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
    "species_threshold": "species.threshold",
    "species_include": "species.include",
    "species_exclude": "species.exclude",
    "detector_threshold": "detector.threshold",
    "visits_gap_seconds": "visits.gap_seconds",
    "lat": "site.lat",
    "lon": "site.lon",
    "margin_minutes": "daylight.margin_minutes",
    "non_bird_days": "retention.non_bird_days",
    "no_alert": "telegram.no_alert",
    "cooldown_minutes": "telegram.cooldown_minutes",
    "publish_min_confidence": "publish.min_confidence",
    "per_species_per_day": "publish.per_species_per_day",
    "dslr_enabled": "dslr.enabled",
    "dslr_auto_shutter": "dslr.auto_shutter",
    "dslr_region": "dslr.region",
    "dslr_daily_cap": "dslr.daily_cap",
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
    species_threshold: float
    species_include: tuple[str, ...]
    species_exclude: tuple[str, ...]
    detector_threshold: float
    visits_gap_seconds: int
    lat: float | None
    lon: float | None
    margin_minutes: int
    non_bird_days: int
    no_alert: tuple[str, ...]
    cooldown_minutes: int
    publish_min_confidence: float
    per_species_per_day: int
    dslr_enabled: bool
    dslr_region: tuple[float, float, float, float]
    dslr_daily_cap: int
    dslr_auto_shutter: bool


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
        "species": {
            "model": str(config.species.model),
            "bank": str(config.species.bank),
            "threshold": config.species.threshold,
            "region_radius_km": config.species.region_radius_km,
            "include": list(config.species.include),
            "exclude": list(config.species.exclude),
        },
        "detector": {
            "model": str(config.detector.model),
            "labels": str(config.detector.labels),
            "threshold": config.detector.threshold,
        },
        "visits": {"gap_seconds": config.visits.gap_seconds},
        "retention": {"non_bird_days": config.retention.non_bird_days},
        "storage": {"data_dir": str(config.storage.data_dir)},
        "web": {"host": config.web.host, "port": config.web.port},
        "telegram": {
            "enabled": config.telegram.enabled,
            "bot_token": config.telegram.bot_token,
            "chat_id": config.telegram.chat_id,
            "base_url": config.telegram.base_url,
            "no_alert": list(config.telegram.no_alert),
            "cooldown_minutes": config.telegram.cooldown_minutes,
        },
        "publish": {
            "enabled": config.publish.enabled,
            "host": config.publish.host,
            "user": config.publish.user,
            "remote_dir": config.publish.remote_dir,
            "site_id": config.publish.site_id,
            "ssh_key": str(config.publish.ssh_key),
            "min_confidence": config.publish.min_confidence,
            "per_species_per_day": config.publish.per_species_per_day,
        },
        "dslr": {
            "enabled": config.dslr.enabled,
            "auto_shutter": config.dslr.auto_shutter,
            "region": list(config.dslr.region),
            "daily_cap": config.dslr.daily_cap,
        },
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
    data["species"]["threshold"] = patch.species_threshold
    data["species"]["include"] = list(patch.species_include)
    data["species"]["exclude"] = list(patch.species_exclude)
    data["detector"]["threshold"] = patch.detector_threshold
    data.setdefault("visits", {})["gap_seconds"] = patch.visits_gap_seconds
    data["site"].pop("lat", None)
    data["site"].pop("lon", None)
    # Write whichever is given; load_config rejects one-without-the-other with a clear message.
    if patch.lat is not None:
        data["site"]["lat"] = patch.lat
    if patch.lon is not None:
        data["site"]["lon"] = patch.lon
    data["daylight"]["margin_minutes"] = patch.margin_minutes
    data["retention"]["non_bird_days"] = patch.non_bird_days
    data["telegram"].update(
        no_alert=list(patch.no_alert),
        cooldown_minutes=patch.cooldown_minutes,
    )
    data["publish"].update(
        min_confidence=patch.publish_min_confidence,
        per_species_per_day=patch.per_species_per_day,
    )
    data.setdefault("dslr", {}).update(
        enabled=patch.dslr_enabled, region=list(patch.dslr_region), daily_cap=patch.dslr_daily_cap,
        auto_shutter=patch.dslr_auto_shutter,
    )
    return data


def save_settings(path: Path, current: Config, patch: SettingsPatch) -> Config:
    """Validate `current` + `patch`, write `path` atomically, return the new Config.

    On any failure the original file is untouched and the temp file is removed.
    """
    # Resolve so a symlinked birdy.toml has its target rewritten, not the link replaced.
    path = Path(path).resolve()
    data = apply_patch(config_to_dict(current), patch)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # holds the bot token
        with os.fdopen(fd, "w") as f:
            f.write(HEADER + tomli_w.dumps(data))
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copymode(path, tmp)
        new = load_config(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return new


_TYPE_WORDS = {int: "an integer", float: "a number"}


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
        raise ConfigError(f"{key} must be {_TYPE_WORDS[cast]}") from None


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


def _lines(form: Mapping[str, str], field: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in form.get(field, "").splitlines() if s.strip())


def _region(form: Mapping[str, str]) -> tuple[float, float, float, float]:
    parts = form.get("dslr_region", "").replace(",", " ").split()
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        vals = ()
    if len(vals) != 4:
        raise ConfigError(f"{FIELD_KEYS['dslr_region']} must be four numbers: x, y, w, h")
    return vals


def parse_settings_form(form: Mapping[str, str]) -> SettingsPatch:
    """Turn the POSTed form into a SettingsPatch; raises ConfigError naming the config key."""
    return SettingsPatch(
        roi=_roi(form),
        pixel_threshold=_num(form, "pixel_threshold", int),
        min_area=_num(form, "min_area", int),
        background_alpha=_num(form, "background_alpha", float),
        cooldown_seconds=_num(form, "cooldown_seconds", float),
        classifier_threshold=_num(form, "classifier_threshold", float),
        species_threshold=_num(form, "species_threshold", float),
        species_include=_lines(form, "species_include"),
        species_exclude=_lines(form, "species_exclude"),
        detector_threshold=_num(form, "detector_threshold", float),
        visits_gap_seconds=_num(form, "visits_gap_seconds", int),
        lat=_num(form, "lat", float, optional=True),
        lon=_num(form, "lon", float, optional=True),
        margin_minutes=_num(form, "margin_minutes", int),
        non_bird_days=_num(form, "non_bird_days", int),
        no_alert=_lines(form, "no_alert"),
        cooldown_minutes=_num(form, "cooldown_minutes", int),
        publish_min_confidence=_num(form, "publish_min_confidence", float),
        per_species_per_day=_num(form, "per_species_per_day", int),
        dslr_enabled=form.get("dslr_enabled") is not None,
        dslr_auto_shutter=form.get("dslr_auto_shutter") is not None,
        dslr_region=_region(form),
        dslr_daily_cap=_num(form, "dslr_daily_cap", int),
    )
