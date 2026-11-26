import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.dslr import (
    EXPOSURE_K_RANGE, TARGET_BRIGHTNESS, Dslr, DslrError, adapt_exposure_k, box_in_frame, dslr_rel, frame_brightness,
    pick_shutter, region_crop, search_tiles, species_crop,
)
from birdy.store import Store

NOW = datetime(2026, 11, 24, 12, 0).astimezone()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


class Runner:
    """Stands in for subprocess.run: records calls and plays a scripted gphoto2."""

    def __init__(self, returncode=0, stdout="", stderr="", write=b"jpegbytes", raises=None):
        self.returncode, self.stdout, self.stderr, self.write, self.raises = returncode, stdout, stderr, write, raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(cmd)
        if self.raises is not None:
            raise self.raises
        if "--filename" in cmd and self.write is not None:
            Path(cmd[cmd.index("--filename") + 1]).write_bytes(self.write)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def make(store, live, runner):
    return Dslr(store, live, runner=runner, now=lambda: NOW)


def tmp_files(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_capture_writes_the_file_atomically(store, live, tmp_path):
    r = Runner()
    dest = tmp_path / "a_eos.jpg"
    took = make(store, live, r).capture(dest)
    assert dest.read_bytes() == b"jpegbytes" and took >= 0 and not tmp_files(tmp_path)
    assert r.calls[0][:2] == ["gphoto2", "--capture-image-and-download"] and "--force-overwrite" in r.calls[0]


@pytest.mark.parametrize("runner, state", [
    (Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n"), "not_connected"),
    (Runner(returncode=1, stderr="*** Error ***\nCould not claim the USB device\n"), "error"),
    (Runner(returncode=0, write=None), "error"),                       # exit 0 but no file
    (Runner(returncode=0, write=b""), "error"),                        # empty file
    (Runner(raises=subprocess.TimeoutExpired("gphoto2", 15)), "error"),
    (Runner(raises=FileNotFoundError("gphoto2")), "error"),
])
def test_capture_failures_raise_and_leave_nothing(store, live, tmp_path, runner, state):
    d = make(store, live, runner)
    dest = tmp_path / "a_eos.jpg"
    with pytest.raises(DslrError):
        d.capture(dest)
    assert not dest.exists() and not tmp_files(tmp_path)
    assert d.status().state == state


def test_daily_cap_blocks_before_gphoto2_runs(store, live, tmp_path):
    for i in range(live.get().dslr.daily_cap):
        when = NOW - timedelta(minutes=1 + i)
        abs_path, rel = store.photo_path_for(when)
        cv2.imwrite(str(abs_path), np.zeros((6, 8, 3), np.uint8))
        cid = store.add_capture(when, rel, 1)
        store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None,
                            dslr_path=str(rel), dslr_box=None)
    r = Runner()
    d = make(store, live, r)
    with pytest.raises(DslrError, match="daily cap"):
        d.capture(tmp_path / "a_eos.jpg")
    assert r.calls == [] and d.status().state == "cap_reached"
    assert d.status().shots_today == live.get().dslr.daily_cap


def test_one_warning_per_distinct_error_and_recovery_is_logged(store, live, tmp_path, caplog):
    bad = Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n")
    d = make(store, live, bad)
    with caplog.at_level(logging.INFO, logger="birdy.dslr"):
        for _ in range(3):
            with pytest.raises(DslrError):
                d.capture(tmp_path / "a_eos.jpg")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
        d._runner = Runner()
        d.capture(tmp_path / "a_eos.jpg")
        assert any("ready" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert d.status().state == "ready"


INFO_OUT = """Label: Capture Target
Readonly: 0
Type: RADIO
Current: Internal RAM
Choice: 0 Internal RAM
END
Label: Auto Power Off
Readonly: 0
Type: TEXT
Current: 0
END
"""


def test_info_parses_current_values(store, live):
    r = Runner(stdout=INFO_OUT)
    d = make(store, live, r)
    assert d.info(("capturetarget", "autopoweroff")) == {"capturetarget": "Internal RAM", "autopoweroff": "0"}
    assert r.calls[0] == ["gphoto2", "--get-config", "capturetarget", "--get-config", "autopoweroff"]
    assert d.status().state == "ready"


def test_info_failure(store, live):
    d = make(store, live, Runner(returncode=1, stderr="*** Error: No camera found. ***\n"))
    with pytest.raises(DslrError):
        d.info()
    assert d.status().state == "not_connected"


def test_dslr_rel():
    assert dslr_rel(Path("2026/11/24/101502_123.jpg")) == Path("2026/11/24/101502_123_eos.jpg")


def test_region_crop_and_box_mapping():
    frame = np.zeros((200, 400, 3), np.uint8)
    region = (0.25, 0.5, 0.5, 0.5)
    assert region_crop(frame, region).shape[:2] == (100, 200)
    # A box in the middle of the region, as fractions of the region -> fractions of the frame.
    assert box_in_frame((0.5, 0.5, 0.2, 0.2), region) == pytest.approx((0.5, 0.75, 0.1, 0.1))


def test_species_crop_is_a_square_twice_the_box():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    crop = species_crop(frame, (0.5, 0.5, 0.1, 0.05))        # box 300 x 100 px -> side 600
    assert crop.shape[:2] == (600, 600)


def test_species_crop_has_a_minimum_and_stays_in_the_frame():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    assert species_crop(frame, (0.0, 0.0, 0.01, 0.01)).shape[:2] == (224, 224)   # tiny box in the corner
    assert species_crop(frame, (0.1, 0.1, 0.8, 0.8)).shape[:2] == (2000, 2000)   # capped by the frame height


def test_a_failed_save_raises_dslr_error_and_leaves_nothing(store, live, tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("birdy.dslr.os.replace", boom)
    d = make(store, live, Runner())
    dest = tmp_path / "a_eos.jpg"
    with pytest.raises(DslrError, match="could not save the frame"):
        d.capture(dest)
    assert not dest.exists() and not tmp_files(tmp_path) and d.status().state == "error"


def test_capture_sets_the_shutter_speed_in_the_same_call(store, live, tmp_path):
    r = Runner()
    make(store, live, r).capture(tmp_path / "a_eos.jpg", "1/250")
    assert r.calls[0][:4] == ["gphoto2", "--set-config", "shutterspeed=1/250", "--capture-image-and-download"]
    # only the one-off questions to a camera that just connected follow: what it is, what it can do
    assert r.calls[1:] == [["gphoto2", "--auto-detect"], ["gphoto2", "--abilities"], ["gphoto2", "--get-config", "shutterspeed"]]


@pytest.mark.parametrize("lux, k, expected", [
    (None, 1.5, None),          # no light reading: leave the camera's own speed
    (0, 1.5, None),
    (5000, 1.5, "1/500"),       # bright: never faster than 1/500
    (750, 1.5, "1/500"),        # needs exactly 1/500
    (500, 1.5, "1/320"),        # needs 1/333: the next slower step that is long enough
    (300, 1.5, "1/200"),
    (100, 1.5, "1/60"),         # needs 1/67
    (10, 1.5, "1/60"),          # dusk: never slower than 1/60
    (750, 3.0, "1/250"),        # a larger constant asks for more time
])
def test_pick_shutter(lux, k, expected):
    assert pick_shutter(lux, k) == expected


def test_frame_brightness_reads_the_region_only():
    frame = np.zeros((160, 320, 3), np.uint8)
    frame[:, 160:] = 200
    assert frame_brightness(frame, (0.5, 0.0, 0.5, 1.0)) == 200
    assert frame_brightness(frame, (0.0, 0.0, 0.5, 1.0)) == 0
    assert frame_brightness(frame, (0.0, 0.0, 1.0, 1.0)) == 100


def test_adapt_exposure_k():
    t = TARGET_BRIGHTNESS
    assert adapt_exposure_k(1.0, t / 2, "1/250") == pytest.approx(2.0)       # half as bright: twice the time
    assert adapt_exposure_k(1.0, 2, "1/250") == pytest.approx(4.0)           # at most 4x per frame
    assert adapt_exposure_k(1.0, t / 2, "1/60") == 1.0                       # slowest speed: a dark frame is expected
    assert adapt_exposure_k(1.0, t, "1/250") == pytest.approx(0.95)          # bright enough: creep faster
    assert adapt_exposure_k(1.0, t * 0.8, "1/250") == pytest.approx(0.95)    # inside the tolerance counts as bright enough
    assert adapt_exposure_k(1.0, t, "1/500") == 1.0                          # already the fastest: nothing to learn
    assert adapt_exposure_k(1.0, t / 2, "1/500") == pytest.approx(2.0)
    assert adapt_exposure_k(50.0, 2, "1/250") == EXPOSURE_K_RANGE[1]         # clamped
    assert adapt_exposure_k(0.0205, t, "1/250") == EXPOSURE_K_RANGE[0]


def test_search_tiles_cover_the_region_with_overlap():
    region = (0.206, 0.346, 0.606, 0.641)                      # the Pi's region on a Large frame
    tiles = search_tiles((3168, 4752, 3), region)
    assert len(tiles) == 12                                    # 4 across x 3 down
    x0, y0, x1, y1 = 0.206 * 4752, 0.346 * 3168, 0.812 * 4752, 0.987 * 3168
    for x, y, w, h in tiles:
        assert round(w * 4752) == 960 and round(h * 3168) == 960
        assert x * 4752 >= x0 - 1 and y * 3168 >= y0 - 1
        assert (x + w) * 4752 <= x1 + 1 and (y + h) * 3168 <= y1 + 1
    xs = sorted({round(t[0] * 4752) for t in tiles})
    ys = sorted({round(t[1] * 3168) for t in tiles})
    assert xs[0] == int(x0) and xs[-1] + 960 == round(x1)       # edge to edge
    assert ys[0] == int(y0) and ys[-1] + 960 == round(y1)
    assert all(b - a <= 960 * 0.75 for a, b in zip(xs, xs[1:]))  # neighbours share >= 25 %
    assert all(b - a <= 960 * 0.75 for a, b in zip(ys, ys[1:]))


def test_search_tiles_of_a_small_or_narrow_region():
    assert search_tiles((800, 900, 3), (0.0, 0.0, 1.0, 1.0)) == []       # fits one tile: search it whole
    tiles = search_tiles((800, 1200, 3), (0.0, 0.0, 1.0, 1.0))           # wide, not tall
    assert tiles == [pytest.approx((0.0, 0.0, 0.8, 1.0)), pytest.approx((0.2, 0.0, 0.8, 1.0))]


# --- model name ---------------------------------------------------------------

AUTO_DETECT = """Model                          Port
----------------------------------------------------------
Canon EOS 50D                  usb:001,005
"""


def test_parse_auto_detect():
    from birdy.dslr import parse_auto_detect
    assert parse_auto_detect(AUTO_DETECT) == "Canon EOS 50D"
    assert parse_auto_detect("Model                          Port\n" + "-" * 58 + "\n") is None
    assert parse_auto_detect("") is None


def test_short_model():
    from birdy.dslr import short_model
    assert short_model("Canon EOS 50D") == "50D"
    assert short_model("Canon EOS 5D Mark II") == "5D Mark II"
    assert short_model("Nikon DSC D7500") == "D7500"
    assert short_model("Sony Alpha-A6000 (Control)") == "A6000"
    assert short_model("GoPro") == "GoPro"
    assert short_model(None) is None and short_model("  ") is None


def test_model_is_read_when_the_camera_answers_and_kept_in_meta(store, live, tmp_path, caplog):
    d = make(store, live, Runner(stdout=AUTO_DETECT))
    assert d.model is None and d.name == "extra camera"
    with caplog.at_level(logging.INFO, logger="birdy.dslr"):
        d.capture(tmp_path / "a_eos.jpg")
    assert d.model == "Canon EOS 50D" and d.name == "50D"
    assert store.get_meta("dslr_model") == "Canon EOS 50D"
    assert "50D ready" in caplog.text
    assert make(store, live, Runner()).model == "Canon EOS 50D"      # known at the next start, camera off


def test_model_lookup_failure_does_not_fail_the_shot(store, live, tmp_path):
    d = make(store, live, Runner(stdout="nothing useful"))
    d.capture(tmp_path / "a_eos.jpg")
    assert d.status().state == "ready" and d.model is None


# --- what the connected camera can do -----------------------------------------------

ABILITIES = """Abilities for camera             : Canon EOS 50D
Serial port support              : no
USB support                      : yes
Capture choices                  :
                                 : Image
                                 : Preview
Configuration support            : yes
"""
NO_CAPTURE = """Abilities for camera             : Some Compact
USB support                      : yes
Capture choices                  : Capture not supported by the driver
Configuration support            : no
"""
SHUTTER_TV = """Label: Shutter Speed
Readonly: 0
Type: RADIO
Current: 1/250
Choice: 0 30
Choice: 1 1/250
Choice: 2 1/500
END
"""
SHUTTER_AV = """Label: Shutter Speed
Readonly: 0
Type: RADIO
Current: auto
Choice: 0 auto
END
"""


def test_parse_can_capture():
    from birdy.dslr import parse_can_capture
    assert parse_can_capture(ABILITIES) is True
    assert parse_can_capture(NO_CAPTURE) is False
    assert parse_can_capture("Capture choices                  : Preview\n") is False
    assert parse_can_capture("nothing here") is None and parse_can_capture("") is None


def test_parse_shutter_support():
    from birdy.dslr import parse_shutter_support
    done = lambda rc, out: subprocess.CompletedProcess([], rc, out, "")  # noqa: E731
    assert parse_shutter_support(done(0, SHUTTER_TV)) == "ok"
    assert parse_shutter_support(done(0, SHUTTER_AV)) == "mode"          # the dial is not on Tv
    assert parse_shutter_support(done(0, SHUTTER_TV.replace("Readonly: 0", "Readonly: 1"))) == "mode"
    assert parse_shutter_support(done(1, "")) == "no"                    # no such setting on this camera


class Scripted:
    """A gphoto2 that answers each question with its own output."""

    def __init__(self, abilities=ABILITIES, shutter=(0, SHUTTER_TV)):
        self.abilities, self.shutter, self.calls = abilities, shutter, []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(cmd)
        if "--filename" in cmd:
            Path(cmd[cmd.index("--filename") + 1]).write_bytes(b"jpeg")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "--auto-detect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, AUTO_DETECT, "")
        if "--abilities" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.abilities, "")
        if cmd[1:] == ["--get-config", "shutterspeed"]:
            return subprocess.CompletedProcess(cmd, self.shutter[0], self.shutter[1], "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_a_connecting_camera_is_asked_what_it_can_do(store, live, tmp_path):
    d = make(store, live, Scripted())
    assert d.status().can_capture is None and d.status().shutter is None and d.accepts_shutter
    d.capture(tmp_path / "a_eos.jpg")
    st = d.status()
    assert st.can_capture is True and st.shutter == "ok" and d.accepts_shutter


def test_a_camera_off_tv_is_fired_at_its_own_speed(store, live, tmp_path):
    r = Scripted(shutter=(0, SHUTTER_AV))
    d = make(store, live, r)
    d.capture(tmp_path / "a_eos.jpg")                  # connects; the probe finds the dial off Tv
    assert d.status().shutter == "mode" and not d.accepts_shutter
    d.capture(tmp_path / "b_eos.jpg", "1/250")
    assert "--set-config" not in r.calls[-1] and (tmp_path / "b_eos.jpg").exists()   # the shot is not lost
    r.shutter = (0, SHUTTER_TV)                        # the dial is turned to Tv; a test frame asks again
    d.probe()
    d.capture(tmp_path / "c_eos.jpg", "1/250")
    assert r.calls[-1][1:3] == ["--set-config", "shutterspeed=1/250"]


def test_a_camera_without_the_setting_or_that_cannot_be_fired(store, live, tmp_path):
    d = make(store, live, Scripted(abilities=NO_CAPTURE, shutter=(1, "")))
    d.probe()
    st = d.status()
    assert st.can_capture is False and st.shutter == "no" and not d.accepts_shutter
