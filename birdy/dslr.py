"""The second camera: a DSLR (Canon EOS 50D) driven with the `gphoto2` command over USB.

The only module that runs gphoto2. `Dslr` takes one frame per call; the pure helpers below turn
a detector box in the DSLR frame into the crop the species model sees."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from birdy.live import LiveConfig
from birdy.store import Store

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15
MIN_SPECIES_SIDE = 224
INFO_KEYS = ("capturetarget", "imageformat", "autopoweroff", "focusmode", "autoexposuremode", "iso", "shuttercounter")

NormBox = tuple[float, float, float, float]

# Auto shutter: speeds birdy may set (seconds, gphoto2's name), fastest first. Faster than 1/500 buys
# nothing for a perched bird; slower than 1/60 blurs it.
SHUTTER_LADDER = (
    (1 / 500, "1/500"), (1 / 400, "1/400"), (1 / 320, "1/320"), (1 / 250, "1/250"), (1 / 200, "1/200"),
    (1 / 160, "1/160"), (1 / 125, "1/125"), (1 / 100, "1/100"), (1 / 80, "1/80"), (1 / 60, "1/60"),
)
# Exposure constant: the shutter time a well-exposed frame needs is K / lux (lens wide open, Auto ISO
# at its top). The default is a first guess; every frame taken corrects it (`adapt_exposure_k`).
EXPOSURE_K_DEFAULT = 1.5
EXPOSURE_K_RANGE = (0.02, 60.0)
TARGET_BRIGHTNESS = 100.0     # mean grey level (0-255) of a well-exposed feeder region


class DslrError(Exception):
    pass


@dataclass(frozen=True)
class DslrStatus:
    state: str            # 'ready' | 'not_connected' | 'error' | 'cap_reached' | 'unknown'
    shots_today: int
    last_error: str | None


def _message(proc: subprocess.CompletedProcess) -> str:
    """gphoto2's most telling output line: the first one naming an error, else the last one."""
    lines = [ln.strip() for ln in ((proc.stderr or "") + "\n" + (proc.stdout or "")).splitlines() if ln.strip()]
    for ln in lines:
        if "error" in ln.casefold() and ln.strip("* ").casefold() != "error":
            return ln.strip("* ")
    return lines[-1].strip("* ") if lines else f"gphoto2 exit {proc.returncode}"


def _state_for(message: str) -> str:
    m = message.casefold()
    return "not_connected" if "no camera found" in m or "could not detect any camera" in m else "error"


class Dslr:
    def __init__(
        self,
        store: Store,
        live: LiveConfig,
        runner: Callable = subprocess.run,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._store = store
        self._live = live
        self._runner = runner
        self._now = now
        self._state = "unknown"
        self._last_error: str | None = None
        self._warned: set[str] = set()

    # --- gphoto2 ----------------------------------------------------------
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            return self._runner(["gphoto2", *args], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise DslrError(f"gphoto2 timed out after {TIMEOUT_SECONDS} s") from None
        except OSError as e:
            raise DslrError(f"gphoto2 could not be run: {e}") from None

    def _fail(self, state: str, message: str) -> None:
        self._state, self._last_error = state, message
        if message not in self._warned:       # one warning per distinct failure, not per bird
            self._warned.add(message)
            log.warning("50D: %s", message)
        raise DslrError(message)

    def _ok(self) -> None:
        if self._state != "ready":
            log.info("50D ready")
        self._state, self._last_error = "ready", None
        self._warned.clear()

    def _midnight(self) -> datetime:
        return self._now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)

    # --- public -----------------------------------------------------------
    def capture(self, dest: Path, shutter: str | None = None) -> float:
        """Take one frame and download it to `dest` (written through a temp name). Returns the
        seconds it took; raises DslrError (camera off, cap reached, timeout...), leaving no file.
        `shutter` (e.g. "1/250") is set on the camera in the same gphoto2 call, before the shot."""
        cap = self._live.get().dslr.daily_cap
        midnight = self._midnight()
        if self._store.count_dslr_shots(midnight) >= cap:
            self._fail("cap_reached", f"daily cap reached ({cap} shots on {midnight:%d/%m}); Pi photos only until midnight")
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".tmp")
        t0 = time.monotonic()
        try:
            setting = ["--set-config", f"shutterspeed={shutter}"] if shutter else []
            proc = self._run([*setting, "--capture-image-and-download", "--filename", str(tmp), "--force-overwrite"])
            # gphoto2 can exit 0 after printing an error: the file is the proof.
            if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                raise DslrError(_message(proc))
            os.replace(tmp, dest)
        except OSError as e:                  # disk full, the day folder gone: the frame is lost, the capture is not
            tmp.unlink(missing_ok=True)
            self._fail("error", f"could not save the frame: {e}")
        except DslrError as e:
            tmp.unlink(missing_ok=True)
            self._fail(_state_for(str(e)), str(e))
        self._ok()
        return time.monotonic() - t0

    def info(self, keys: tuple[str, ...] = INFO_KEYS) -> dict[str, str]:
        """The camera's current value for each config key, in one gphoto2 call."""
        args: list[str] = []
        for k in keys:
            args += ["--get-config", k]
        try:
            proc = self._run(args)
            if proc.returncode != 0:
                raise DslrError(_message(proc))
        except DslrError as e:
            self._fail(_state_for(str(e)), str(e))
        out: dict[str, str] = {}
        blocks = [b for b in (proc.stdout or "").split("END") if b.strip()]
        for key, block in zip(keys, blocks):
            for line in block.splitlines():
                if line.startswith("Current:"):
                    out[key] = line.split(":", 1)[1].strip()
        self._ok()
        return out

    def status(self) -> DslrStatus:
        shots = self._store.count_dslr_shots(self._midnight())
        state = self._state
        if state == "cap_reached" and shots < self._live.get().dslr.daily_cap:
            state = "ready"                   # a new day, or the cap was raised
        return DslrStatus(state=state, shots_today=shots, last_error=self._last_error)


# --- geometry (pure) --------------------------------------------------------

def dslr_rel(pi_rel: Path) -> Path:
    """Where a capture's DSLR frame lives: beside the Pi photo, `_eos` before the extension."""
    pi_rel = Path(pi_rel)
    return pi_rel.with_name(f"{pi_rel.stem}_eos{pi_rel.suffix}")


def region_crop(frame: np.ndarray, region: NormBox) -> np.ndarray:
    """The part of the frame the detector searches; `region` is (x, y, w, h) in fractions."""
    h, w = frame.shape[:2]
    x, y, rw, rh = region
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = min(w, max(x0 + 1, round((x + rw) * w))), min(h, max(y0 + 1, round((y + rh) * h)))
    return frame[y0:y1, x0:x1]


TILE_PX = 960          # the detector sees 320 px: a bird must be ~1/10 of what it is shown
TILE_OVERLAP = 0.25    # of a tile, so a bird on a seam is whole in one of them


def search_tiles(frame_shape: tuple[int, ...], region: NormBox, tile_px: int = TILE_PX,
                 overlap: float = TILE_OVERLAP) -> list[NormBox]:
    """Overlapping squares of about `tile_px` covering `region`, as (x, y, w, h) fractions of
    the frame, row by row. Empty when the region already fits one tile (search it whole)."""
    fh, fw = frame_shape[:2]
    x, y, rw, rh = region
    x0, y0 = int(x * fw), int(y * fh)
    w, h = min(fw, round((x + rw) * fw)) - x0, min(fh, round((y + rh) * fh)) - y0
    if w <= tile_px and h <= tile_px:
        return []

    def starts(length: int) -> tuple[list[int], int]:
        size = min(tile_px, length)
        if length <= size:
            return [0], size
        n = 1 + -(-(length - size) // max(1, int(size * (1 - overlap))))
        return [round(i * (length - size) / (n - 1)) for i in range(n)], size

    xs, tw = starts(w)
    ys, th = starts(h)
    return [((x0 + tx) / fw, (y0 + ty) / fh, tw / fw, th / fh) for ty in ys for tx in xs]


def box_in_frame(box: NormBox, region: NormBox) -> NormBox:
    """A box given as fractions of `region` -> fractions of the whole frame."""
    bx, by, bw, bh = box
    rx, ry, rw, rh = region
    return (rx + bx * rw, ry + by * rh, bw * rw, bh * rh)


def species_crop(frame: np.ndarray, box: NormBox) -> np.ndarray:
    """A square around the bird: side = 2 x the box's longer side (at least MIN_SPECIES_SIDE px),
    capped by the frame and shifted to stay inside it. `box` is in fractions of the frame."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = box
    side = max(2 * max(bw * w, bh * h), MIN_SPECIES_SIDE)
    side = int(min(round(side), w, h))
    cx, cy = (bx + bw / 2) * w, (by + bh / 2) * h
    x0 = int(min(max(round(cx - side / 2), 0), w - side))
    y0 = int(min(max(round(cy - side / 2), 0), h - side))
    return frame[y0:y0 + side, x0:x0 + side]


# --- auto shutter (pure) ----------------------------------------------------

def pick_shutter(lux: float | None, k: float) -> str | None:
    """The fastest ladder speed that still exposes the frame at this light level (time >= k / lux);
    the slowest one when even that is too fast. None when there is no light reading."""
    if lux is None or lux <= 0:
        return None
    needed = k / lux
    for seconds, name in SHUTTER_LADDER:
        if seconds >= needed:
            return name
    return SHUTTER_LADDER[-1][1]


def frame_brightness(frame: np.ndarray, region: NormBox) -> float:
    """Mean grey level (0-255) of the feeder region, on a subsampled frame."""
    return float(region_crop(frame, region)[::8, ::8].mean())


def adapt_exposure_k(k: float, brightness: float, shutter: str) -> float:
    """Correct the exposure constant from a frame taken at `shutter`.

    Too dark (< 75 % of the target): the frame needed more time, so k grows by the shortfall (at
    most 4x per frame) - unless the shutter was already the slowest allowed, where a dark frame is
    expected and says nothing about k. Bright enough: k shrinks 5 %, so the speed creeps back up
    until a frame comes out a little dark - unless the shutter was already the fastest."""
    lo, hi = EXPOSURE_K_RANGE
    if brightness < 0.75 * TARGET_BRIGHTNESS:
        if shutter != SHUTTER_LADDER[-1][1]:
            k *= min(TARGET_BRIGHTNESS / max(brightness, 1.0), 4.0)
    elif shutter != SHUTTER_LADDER[0][1]:
        k *= 0.95
    return min(max(k, lo), hi)
